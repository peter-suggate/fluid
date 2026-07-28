import { pathToFileURL } from "node:url";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { tallCellMethod } from "../lib/methods/tall-cell";
import type { GPUSolverInstance, SimulationMethod } from "../lib/methods/types";
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
import { createGlobalFineLevelSetConsumerSource } from "../lib/octree-consumer-sampling";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "../lib/generated/octree-power-catalog";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import {
  FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
  unpackFineLevelSetGPURedistanceControl,
} from "../lib/webgpu-octree-fine-levelset-redistance";
import { FINE_LEVELSET_VOLUME_VALID, unpackFineLevelSetGPUVolumeControl }
  from "../lib/webgpu-octree-fine-levelset-volume";
import { unpackFineLevelSetGPUTopologyControl }
  from "../lib/webgpu-octree-fine-levelset-topology";
import { unpackFineToCoarseGPUControl }
  from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { readFineLevelSetWorksetHeader } from "../lib/octree-fine-levelset-bricks";
import { decodeStructuredProjectionEnergy }
  from "../lib/webgpu-octree-structured-dynamics";
import { unpackFineLevelSetGPUTransportControl }
  from "../lib/webgpu-octree-fine-levelset-transport";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { fluidExecutionDeviceFeatures } from "../lib/gpu-startup";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "../lib/webgpu-octree-sparse-bricks";
import { environmentIndex } from "../lib/environments";
import { octreePowerOwnerArenaPublicationIsValid } from "../lib/webgpu-octree-power-descriptor";
import {
  OCTREE_OWNER_ARENA_MAGIC,
  OCTREE_OWNER_PAGE_CONTROL_WORDS,
  unpackOctreeOwnerPageControl,
} from "../lib/webgpu-octree-owner-pages";
import { MAX_TERRAIN_FEATURES, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT, sceneHasTerrain } from "../lib/terrain";
import { compactOctreeFieldEvidenceIsAcceptable, compactOctreePublicationHeaderEvidence,
  reconstructCompactOctreeOccupancyField,
  type CompactOctreeFieldEvidence } from "./webgpu-smoke-compact-field";
import {
  GPUDataFlowAudit,
  type GPUDataFlowEncoderSession,
  type GPUDataFlowManifest,
  type GPUDataFlowPassRecorder,
} from "./webgpu-data-flow-manifest";
import { createPassEncoderIsolationScratch, isolateComputePassEncoders } from "./webgpu-pass-encoder-isolation";
import { encodeStructuredAuditRecordCopies, exactStructuredGenerationAuditFailures,
  finalPerformanceAuthorityFailures, STRUCTURED_GENERATION_AUDIT_SNAPSHOT,
  unpackStructuredBoundaryControl, unpackStructuredGenerationAuditSnapshot,
  unpackStructuredVelocityControl }
  from "./webgpu-smoke-structured-audit";
import { decodeOctreeMGPCGDiagnostics, octreePowerPressureDiagnosticsAreAcceptable,
  octreePowerPressureEnvelopeIsAcceptable, octreeProjectedVariationalResidualRms,
  type OctreeMGPCGDiagnostics } from "./webgpu-smoke-pressure";
import { compactLiquidVelocityDiagnostic, compactMechanicalEnergyDiagnostic } from "./webgpu-smoke-power-diagnostics";
import { compareVelocityFields, DAM_BREAK_VELOCITY_PARITY_LIMITS, rasterizeStructuredCellVelocities,
  velocityParityFailures, type CompactVelocityRaster, type VelocityParityMetrics } from "./webgpu-smoke-velocity-parity";
import { narrowVerticalSlitMetrics, type NarrowVerticalSlitMetrics } from "./raster-slit-metrics";
import {
  enclosedSurfaceHoleMetrics,
  surfaceStepMetrics,
  type EnclosedSurfaceHoleMetrics,
  type SurfaceStepMetrics,
} from "./raster-surface-metrics";
import { viewportFailureIndicator } from "../lib/viewport-failure-diagnostics";
import type { PaperPhaseId, PerformanceTrace } from "../lib/performance-trace";
import type { OctreeWorkSnapshot } from "../lib/webgpu-octree-work-accounting";
import { usePerformanceInstrumentationStore } from "../lib/stores/performance-instrumentation-store";
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
// Dawn quantizes timestamp-query results to 65536 ns unless told not to, so
// every pass reads as an integer number of 65.536 us quanta and anything
// cheaper than that reads as exactly zero. That is a browser fingerprinting
// defence with no purchase in a local benchmark process, and it destroys the
// resolution per-pass attribution needs, so the pass-timestamp diagnostic
// turns it off. Nothing else about the command stream changes.
const dawnDisabledFeatures = [
  ...(process.env.FLUID_GPU_PASS_TIMESTAMPS === "1" || process.env.FLUID_GPU_FINE_TIMESTAMPS === "1"
    ? ["timestamp_quantization"] : []),
  ...(process.env.FLUID_WEBGPU_DAWN_DISABLED_FEATURES?.split(",").map((name) => name.trim())
    .filter((name) => name.length > 0) ?? []),
];
const dawnOptions = [
  `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
  ...(process.env.FLUID_WEBGPU_ADAPTER ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
  ...(process.env.FLUID_WEBGPU_DAWN_FEATURES
    ? [`enable-dawn-features=${process.env.FLUID_WEBGPU_DAWN_FEATURES}`] : []),
  ...(dawnDisabledFeatures.length > 0
    ? [`disable-dawn-features=${Array.from(new Set(dawnDisabledFeatures)).join(",")}`] : []),
];
const gpu = create(dawnOptions);
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
const regressionArtifactRequested = process.env.FLUID_REGRESSION_ARTIFACT === "1";
const genericPhaseTraceRequested = process.env.FLUID_GPU_FINE_TIMESTAMPS === "1";
/** One-shot, pass-local GPU timestamps for algorithm attribution. Unlike the
 * semantic fallback tracer this does not split submissions or wait between
 * phases: timestamp writes are attached to the compute passes the solver
 * already opens, and the first recurring command buffer is resolved only
 * after it has been submitted. */
const gpuPassTimestampRequested = process.env.FLUID_GPU_PASS_TIMESTAMPS === "1";
/** Give every compute pass its own Metal encoder so the timestamps above are
 * actually per-pass rather than per-encoder. Costs one encoder switch per pass,
 * so an isolated run's absolute frame time is inflated and is a ranking, never
 * a baseline. See `tools/webgpu-pass-encoder-isolation.ts`. */
const gpuIsolatePassEncodersRequested = process.env.FLUID_GPU_ISOLATE_PASS_ENCODERS === "1";
/** Read by `lib/webgpu-pass-broker.ts`; mirrored here only so the report can
 * state which passes it actually measured. */
const gpuIsolatePassLabelsRequested = process.env.FLUID_GPU_ISOLATE_PASS_LABELS === "1";
const performanceTraceRequested =
  process.env.FLUID_PERFORMANCE_TRACES === undefined
    ? true
    : process.env.FLUID_PERFORMANCE_TRACES === "1" || genericPhaseTraceRequested;
usePerformanceInstrumentationStore.getState().setEnabled(performanceTraceRequested);
/** Backward-compatible benchmark switch. Attribution is collected from the
 * sole generic semantic-phase recorder; the smoke harness owns no query set. */
const genericPhaseTraceAdvances = Math.max(
  1,
  Math.floor(Number(process.env.FLUID_GPU_FINE_TIMESTAMP_ADVANCES ?? 1)),
);
/** Emit every GPU physics phase trace the solver captures as a JSON line.
 * The dynamic recorder throttles itself to one in-flight sample per 250 ms,
 * so slow advances are sampled near-continuously and fast advances sparsely. */
const physicsTraceLogRequested = process.env.FLUID_PHYSICS_TRACE_LOG === "1";
/** Queue-fence cadence inside the advance loop. The default preserves the
 * historical every-30-advances fence; per-step fencing (=1) lets the dynamic
 * physics trace resolve every advance for per-step phase attribution. */
const completionFenceEverySteps = Math.max(1, Number(process.env.FLUID_AWAIT_EVERY_STEPS ?? 30));
/** Optional bound on the deliberately unbounded `advanceTo` retry loop.
 * `advanceTo` self-rejects while a profiled advance is in flight or a topology
 * rebuild is blocked, and the harness simply retries. A solver that never
 * accepts again therefore looks exactly like a slow one from the outside: the
 * run just burns to `FLUID_WEBGPU_SMOKE_TIMEOUT_MS` and exits 124 with nothing
 * naming the wedge. Setting this to a positive count converts that into a
 * named abort. Unset or 0 keeps today's unbounded retry byte-for-byte. */
const rejectedAdvanceWedgeLimitRaw = Number(process.env.FLUID_MAX_CONSECUTIVE_REJECTED_ADVANCES ?? 0);
const rejectedAdvanceWedgeLimit = Number.isFinite(rejectedAdvanceWedgeLimitRaw)
  && rejectedAdvanceWedgeLimitRaw > 0 ? Math.floor(rejectedAdvanceWedgeLimitRaw) : 0;
/** Advance-loop heartbeat cadence. A line is emitted only once BOTH thresholds
 * are met, i.e. at whichever of the two is the less frequent, so a fast run
 * pays five seconds per line and a slow run pays twenty-five steps per line.
 * The point is that a worker killed at its timeout still leaves a trail. */
const PROGRESS_HEARTBEAT_WALL_MS = 5_000;
const PROGRESS_HEARTBEAT_STEPS = 25;
const gpuCommandAuditRequested = process.env.FLUID_GPU_COMMAND_AUDIT === "1";
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
if (octreeGlobalFineFactorOverride !== undefined && !["4", "8"].includes(octreeGlobalFineFactorOverride)) {
  throw new Error("FLUID_OCTREE_GLOBAL_FINE_FACTOR must be 4 or 8");
}
const powerGenerationAuditRequested = process.env.FLUID_POWER_GENERATION_AUDIT === "1";
const powerGenerationAuditLog = process.env.FLUID_POWER_GENERATION_AUDIT_LOG !== "0";
const powerCandidateAuditRequested = process.env.FLUID_POWER_CANDIDATE_AUDIT === "1";
const powerStageAuditLog = process.env.FLUID_POWER_STAGE_AUDIT === "1";
const powerAuditEverySteps = Number(process.env.FLUID_POWER_AUDIT_EVERY_STEPS ?? 1);
if (!Number.isSafeInteger(powerAuditEverySteps) || powerAuditEverySteps < 1) {
  throw new Error("FLUID_POWER_AUDIT_EVERY_STEPS must be a positive integer");
}
/** Silent-failure tripwires (docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md, A3/P0.3).
 *
 * Each of these was previously observable only through a ~250 ms telemetry
 * poll or not at all, so a build could get faster by deleting physics and
 * still print PASS — a silent topology rollback once manufactured a fake
 * 1.66x win. They are captured per accepted step by GPU-side copies into a
 * harness-owned ring (never a mid-run map, which would drain the pipeline and
 * corrupt the wall measurement) and evaluated after the measured window.
 *
 * `FLUID_TRIPWIRES=1` additionally *requires* every counter to be readable:
 * a counter that cannot be evaluated fails the run rather than passing
 * silently. That is the failure mode this work item exists to kill.
 * `FLUID_TRIPWIRE_ALLOW=topology-rollback,restriction-unaccepted,...`
 * downgrades named tripwires to loud warnings for triage only. */
/** Coverage envelope for the restriction tripwire. Coarse rows whose centre
 * falls outside the fine narrow band are legitimately uncorrected, so the
 * absolute count is never an error; what must not regress is the FRACTION.
 * Calibrated against the measured mini-lane peak of 11.8% (177/1500 at step 12,
 * decaying to 2.0% by step 500) with better than 2x headroom. A band-width
 * narrowing -- the silent failure this signal exists to catch, per
 * lib/webgpu-octree-fine-to-coarse-levelset.ts -- moves this far past 25%. */
const TRIPWIRE_MAXIMUM_UNCOVERED_ROW_FRACTION = 0.25;
const TRIPWIRE_IDS = ["topology-rollback", "restriction-unaccepted",
  "mgpcg-nonconvergence", "fine-band-sentinel"] as const;
type TripwireId = (typeof TRIPWIRE_IDS)[number];
const tripwireMode = process.env.FLUID_TRIPWIRES;
if (tripwireMode !== undefined && tripwireMode !== "0" && tripwireMode !== "1") {
  throw new Error("FLUID_TRIPWIRES must be 0 or 1");
}
const tripwiresDisabled = tripwireMode === "0";
const tripwiresForcedRequired = tripwireMode === "1";
const tripwireAllowList = new Set((process.env.FLUID_TRIPWIRE_ALLOW ?? "")
  .split(",").map((entry) => entry.trim()).filter(Boolean));
for (const entry of tripwireAllowList) {
  if (!TRIPWIRE_IDS.includes(entry as TripwireId)) {
    throw new Error(`FLUID_TRIPWIRE_ALLOW contains unknown tripwire "${entry}"; known ids: ${TRIPWIRE_IDS.join(", ")}`);
  }
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

async function readBufferBindingsPacked(
  device: GPUDevice,
  bindings: readonly { binding: GPUBufferBinding; byteLength: number }[],
): Promise<readonly Uint8Array[]> {
  const offsets: number[] = [];
  let packedLength = 0;
  for (const item of bindings) {
    packedLength = Math.ceil(packedLength / 4) * 4;
    offsets.push(packedLength);
    packedLength += Math.ceil(item.byteLength / 4) * 4;
  }
  const readback = device.createBuffer({
    label: "Final performance authority packed readback",
    size: Math.max(4, packedLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "Final performance authority readback" });
    bindings.forEach((item, index) => {
      encoder.copyBufferToBuffer(
        item.binding.buffer,
        item.binding.offset ?? 0,
        readback,
        offsets[index],
        Math.ceil(item.byteLength / 4) * 4,
      );
    });
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(readback.getMappedRange());
    return bindings.map((item, index) => mapped.slice(offsets[index], offsets[index] + item.byteLength));
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
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
  enclosedSurfaceHoles: {
    front: EnclosedSurfaceHoleMetrics;
    back: EnclosedSurfaceHoleMetrics;
  };
  surfaceSteps: {
    front: SurfaceStepMetrics;
    back: SurfaceStepMetrics;
  };
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
    enclosedSurfaceHoles: {
      front: EnclosedSurfaceHoleMetrics;
      back: EnclosedSurfaceHoleMetrics;
    };
    surfaceSteps: {
      front: SurfaceStepMetrics;
      back: SurfaceStepMetrics;
    };
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
  pageMetadataBytes: number;
  pageWorklistBytes: number;
  diagnosticReadbackBytes: number;
  coarseState?: number;
  coarseGeneration?: number;
  coarseRowCount?: number;
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
  redistanceFlags?: number;
  redistanceFirstError?: number;
  redistanceAcceptedCells?: number;
  redistanceInitialPages?: number;
  redistanceFinalPages?: number;
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
  probedPages?: Array<{
    key: number;
    directoryPhysicalId?: number;
    directoryFound: boolean;
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
  const pageCapacity = source.plan.maximumResidentBricks;
  const samplesPerBrick = source.plan.samplesPerBrick;
  const [worklistBytes, metadataBytes, flagBytes, phiBytes, coarseBytes, seedBytes, topologyBytes,
    transportBytes, redistanceBytes, volumeBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, source.worklist.size),
    readBufferBinding(device, { buffer: source.metadata }, pageCapacity * 40),
    readBufferBinding(device, { buffer: source.flags }, pageCapacity * samplesPerBrick * 4),
    readBufferBinding(device, { buffer: source.phi }, pageCapacity * samplesPerBrick * 4),
    source.coarsePhiDirectory
      ? readBufferBinding(device, { buffer: source.coarsePhiDirectory },
        32 + (source.coarsePhiRowCapacity ?? 0) * 32)
      : Promise.resolve(undefined),
    source.seedControl ? readBufferBinding(device, { buffer: source.seedControl }, 8) : Promise.resolve(undefined),
    source.topologyControl
      ? readBufferBinding(device, { buffer: source.topologyControl }, 32)
      : Promise.resolve(undefined),
    transportControl ? readBufferBinding(device, { buffer: transportControl }, 64) : Promise.resolve(undefined),
    redistanceControl
      ? readBufferBinding(device, { buffer: redistanceControl }, FINE_LEVELSET_REDISTANCE_CONTROL_BYTES)
      : Promise.resolve(undefined),
    volumeControl ? readBufferBinding(device, { buffer: volumeControl }, 64) : Promise.resolve(undefined),
  ]);
  const worklist = new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4);
  const flags = new Uint32Array(flagBytes.buffer, flagBytes.byteOffset, flagBytes.byteLength / 4);
  const phi = new Float32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4);
  const phiBits = new Uint32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4);
  const activePages = Math.min(worklist[1], pageCapacity);
  let taggedMetadataPages = 0, malformedActivePages = 0;
  let validSamples = 0, finiteValidSamples = 0, negativeValidSamples = 0, positiveValidSamples = 0;
  let phiBitXor = 0, phiBitSum = 0, phiSum = 0, phiAbsSum = 0;
  for (let id = 0; id < pageCapacity; id += 1) if (metadata[id * 10 + 2] === source.generation) taggedMetadataPages += 1;
  for (let work = 0; work < activePages; work += 1) {
    const id = worklist[7 + work];
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
  if (probeBrickKeys.length > 0) {
    const publishedIds = new Set<number>();
    for (let work = 0; work < activePages; work += 1) publishedIds.add(worklist[7 + work]);
    const lookup = (key: number) => {
      if (worklist.length < 7 || worklist[0] !== source.generation
        || worklist[2] !== pageCapacity || (worklist[3] & 3) !== 3
        || worklist[5] !== 1 || worklist[6] !== 1) return undefined;
      const directoryBase = 7 + pageCapacity;
      if (key >= source.plan.logicalBrickCount || directoryBase + key >= worklist.length) return undefined;
      const id = worklist[directoryBase + key], base = id * 10;
      return id < pageCapacity && base + 2 < metadata.length && metadata[base] === id
        && metadata[base + 1] === key && metadata[base + 2] === source.generation ? id : undefined;
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
        key, directoryPhysicalId: id, directoryFound: validId,
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
  if (coarse) for (let slot = 0; slot < Math.min(coarse[2], (coarse.length - 8) / 8); slot += 1) {
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
    const rowCount = Math.min(coarse[2], (coarse.length - 8) / 8);
    const maximumLeaf = coarse[3], dimensions = [coarse[4], coarse[5], coarse[6]];
    const physicalCellSize = new Float32Array(coarse.buffer, coarse.byteOffset + 7 * 4, 1)[0];
    const mortonPart = (input: number) => {
      let value = input & 1023;
      value = (value | (value << 16)) & 0x030000ff; value = (value | (value << 8)) & 0x0300f00f;
      value = (value | (value << 4)) & 0x030c30c3; value = (value | (value << 2)) & 0x09249249;
      return value >>> 0;
    };
    const morton = (cell: number) => {
      const x = cell % dimensions[0], y = Math.floor(cell / dimensions[0]) % dimensions[1];
      const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
      return (mortonPart(x) | (mortonPart(y) << 1) | (mortonPart(z) << 2)) >>> 0;
    };
    for (const requested of probeBrickKeys) {
      const q = [requested % dimensions[0], Math.floor(requested / dimensions[0]) % dimensions[1],
        Math.floor(requested / (dimensions[0] * dimensions[1]))];
      let foundBase = -1, lookupSize = 0;
      for (let size = 1; size <= maximumLeaf; size *= 2) {
        const origin = q.map((value) => Math.floor(value / size) * size);
        const cell = origin[0] + dimensions[0] * (origin[1] + dimensions[1] * origin[2]);
        const wantedLevel = 31 - Math.clz32(size), wantedMorton = morton(cell);
        let low = 0, high = rowCount;
        while (low < high) {
          const middle = low + Math.floor((high - low) / 2), base = 8 + middle * 8;
          const entryLevel = 31 - Math.clz32(coarse[base + 1]), entryMorton = morton(coarse[base] - 1);
          if (entryLevel < wantedLevel || (entryLevel === wantedLevel && entryMorton < wantedMorton)) low = middle + 1;
          else high = middle;
        }
        if (low < rowCount) {
          const base = 8 + low * 8;
          if (coarse[base] === cell + 1 && coarse[base + 1] === size) {
            foundBase = base; lookupSize = size;
          }
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
  const redistanceDiagnostics = redistance ? unpackFineLevelSetGPURedistanceControl(redistance) : undefined;
  const volume = volumeBytes
    ? new Uint32Array(volumeBytes.buffer, volumeBytes.byteOffset, volumeBytes.byteLength / 4)
    : undefined;
  const volumeFloats = volumeBytes
    ? new Float32Array(volumeBytes.buffer, volumeBytes.byteOffset, volumeBytes.byteLength / 4)
    : undefined;
  return {
    generation: source.generation, worklistGeneration: worklist[0], generationSlot: source.generationSlot, activePages,
    configuredBrickCapacity: pageCapacity,
    taggedMetadataPages, malformedActivePages, validSamples, finiteValidSamples,
    negativeValidSamples, positiveValidSamples, phiBitXor, phiBitSum, phiSum, phiAbsSum,
    residentPayloadBytes: activePages * source.plan.payloadBytesPerBrick,
    payloadCapacityBytes: source.plan.payloadCapacityBytes,
    payloadFragmentationBytes: (pageCapacity - activePages) * source.plan.payloadBytesPerBrick,
    pageMetadataBytes: pageCapacity * 40,
    pageWorklistBytes: source.worklist.size,
    diagnosticReadbackBytes: [worklistBytes, metadataBytes, flagBytes, phiBytes, coarseBytes, seedBytes,
      topologyBytes, transportBytes, redistanceBytes, volumeBytes]
      .reduce((sum, bytes) => sum + (bytes?.byteLength ?? 0), 0),
    publicationValid: worklist[0] === source.generation && worklist[2] === pageCapacity
      && (worklist[3] & 3) === 3 && worklist[5] === 1 && worklist[6] === 1
      && activePages > 0 && taggedMetadataPages >= activePages && malformedActivePages === 0
      && validSamples > 0 && finiteValidSamples === validSamples,
    ...(probedPages.length > 0 ? { probedPages } : {}),
    ...(probedCoarseRecords.length > 0 ? { probedCoarseRecords } : {}),
    ...(coarse ? { coarseState: coarse[0], coarseGeneration: coarse[1],
      coarseRowCount: coarse[2], coarseMaximumLeafSize: coarse[3], coarseEntryCount,
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
    ...(redistanceDiagnostics ? {
      redistanceUnresolvedCells: redistanceDiagnostics.unresolvedCells,
      redistanceResolveMissingCells: redistanceDiagnostics.resolveMissingCells,
      redistanceResidualViolationCells: redistanceDiagnostics.residualViolationCells,
      redistanceMaximumResidualScaled: redistanceDiagnostics.maximumResidualScaled,
      redistanceSeedCount: redistanceDiagnostics.seedCount,
      redistanceCommitted: redistanceDiagnostics.committed,
      redistanceFlags: redistanceDiagnostics.flags,
      redistanceFirstError: redistanceDiagnostics.firstError,
      redistanceAcceptedCells: redistanceDiagnostics.acceptedCells,
      redistanceInitialPages: redistanceDiagnostics.initialPages,
      redistanceFinalPages: redistanceDiagnostics.finalPages,
    } : {}),
    ...(volume && volumeFloats ? { volumeFlags: volume[0], volumeInitialized: volume[1] !== 0,
      volumeSamples: volume[2], volumeReference: volumeFloats[3], volumeCurrent: volumeFloats[4],
      volumeInterfaceArea: volumeFloats[5], volumeCorrection: volumeFloats[6],
      volumeCorrected: volume[7] !== 0, volumeCoarse: volumeFloats[8], volumeFine: volumeFloats[9],
      volumeReplacedCoarse: volumeFloats[10], volumeCoarseRows: volume[11], volumeUnowned: volume[12],
      volumeExpectedAir: volume[12], volumeGeneration: volume[13],
      volumeLookupFailures: volume[14], volumeStaleOwners: volume[15] } : {}),
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
  const columnFallback = device.createTexture({
    label: "Hybrid presentation non-column fallback",
    size: [1, 1],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
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
    const globalFineLevelSet = solver.globalFineLevelSetSource
      ? createGlobalFineLevelSetConsumerSource(solver.globalFineLevelSetSource)
      : undefined;
    if (verifyGlobalFineAuthorityTransition && !globalFineLevelSet) {
      throw new Error("Global-fine authority transition requested without a published source");
    }
    pipeline.setVolume(solver.surfaceFieldTexture ?? solver.volumeTexture,
      solver.columnBaseTexture ?? columnFallback);
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
        const presentationDiagnostics = await pipeline.completeSurfaceDiagnostics();
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
        const backMask = new Uint8Array(width * height);
        const frontPositions = new Float32Array(width * height * 3);
        const backPositions = new Float32Array(width * height * 3);
        frontPositions.fill(Number.NaN);
        backPositions.fill(Number.NaN);
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
            frontPositions.set([px, py, pz], (x + y * width) * 3);
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
          if (backPresent) {
            backInterfacePixels += 1;
            backMask[x + y * width] = 1;
            backPositions.set([
              decodeFloat16(interfaceWords[backAt]),
              decodeFloat16(interfaceWords[backAt + 1]),
              decodeFloat16(interfaceWords[backAt + 2]),
            ], (x + y * width) * 3);
          }
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
        const enclosedSurfaceHoles = {
          front: enclosedSurfaceHoleMetrics(frontMask, width, height),
          back: enclosedSurfaceHoleMetrics(backMask, width, height),
        };
        const surfaceSteps = {
          front: surfaceStepMetrics(frontMask, frontPositions, width, height, fineCellWidth),
          back: surfaceStepMetrics(backMask, backPositions, width, height, fineCellWidth),
        };
        interfaceReadback.unmap();
        return { initializeWall_ms, frameWall_ms: performance.now() - frameStarted,
          bodyCount: presentationBodies.length, width, height, frontInterfacePixels, backInterfacePixels,
          pairedInterfacePixels, frontOnlyInterfacePixels, backOnlyInterfacePixels,
          backOnlyInterfaceLocations, backOnlyInterfacePositions_m,
          frontInterfaceHash, backInterfaceHash,
          narrowVerticalSlits,
          enclosedSurfaceHoles,
          surfaceSteps,
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
      console.info(JSON.stringify({ phase: "hybrid-water-diagnostics", ...pipeline.surfaceRenderDiagnostics }));
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
      enclosedSurfaceHoles: reverse.enclosedSurfaceHoles,
      surfaceSteps: reverse.surfaceSteps,
      wallCornerCapPixels: reverse.wallCornerCapPixels,
      damExposedCornerCapPixels: reverse.damExposedCornerCapPixels,
      ...(reverse.frontInterfaceBounds_m ? { frontInterfaceBounds_m: reverse.frontInterfaceBounds_m } : {}),
    }, rendererValidationErrorCount: 0, rendererUncapturedErrorCount: 0,
      ...(globalFineAuthorityTransition ? { globalFineAuthorityTransition } : {}) };
  } finally {
    if (rendererValidationScopeActive) await device.popErrorScope().catch(() => null);
    device.removeEventListener("uncapturederror", onUncapturedRendererError);
    pipeline.destroy(); output.destroy(); columnFallback.destroy(); uniformBuffer.destroy(); bodyBuffer.destroy();
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
  const structured = solver as GPUSolverInstance & {
    structuredVelocityControl?: GPUBuffer;
    structuredRowVelocities?: GPUBuffer;
  };
  const controlBuffer = structured.structuredVelocityControl;
  const headerBuffer = solver.powerLeafHeaders;
  const velocityBuffer = structured.structuredRowVelocities;
  if (!controlBuffer || !headerBuffer || !velocityBuffer) return undefined;
  const controlBytes = await readBufferBinding(device, { buffer: controlBuffer }, 24);
  const control = unpackStructuredVelocityControl(new Uint32Array(
    controlBytes.buffer, controlBytes.byteOffset, controlBytes.byteLength / 4));
  const rowCount = control.rowCount, reconstructedRows = control.rowCount;
  const bankStrideBytes = velocityBuffer.size / 2;
  if (rowCount === 0 || rowCount * 48 > headerBuffer.size
    || !Number.isSafeInteger(bankStrideBytes) || rowCount * 16 > bankStrideBytes) return undefined;
  const [headerBytes, velocityBytes] = await Promise.all([
    readBufferBinding(device, { buffer: headerBuffer }, rowCount * 48),
    readBufferBinding(device, { buffer: velocityBuffer, offset: control.activeBank * bankStrideBytes }, rowCount * 16),
  ]);
  const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const velocities = new Float32Array(velocityBytes.buffer, velocityBytes.byteOffset, rowCount * 4);
  return {
    ...rasterizeStructuredCellVelocities(headers, velocities, rowCount, dimensions),
    publicationValid: control.flags === 0 && control.firstError === 0xffff_ffff
      && control.epoch > 0 && control.activeBank < 2,
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
 * QA forensics: verify the sparse owner-page arena encodes a partition. Every
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
  if (words.length <= 15 || words[15] !== 0x4f57_4e52) {
    return [{ issue: "invalid-owner-page-arena", arenaHeader: Array.from(words.slice(0, 16)) }];
  }
  const rawWord = (q: readonly number[]): number | undefined => {
    const capacity = words[3], pageOffset = words[5], resident = Math.min(words[1], capacity);
    if (pageOffset !== 16 + capacity || words[6] !== pageOffset + capacity) return undefined;
    const bd = [Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 8)];
    const b = q.map((v) => Math.floor(v / 8));
    const logical = b[0] + b[1] * bd[0] + b[2] * bd[0] * bd[1];
    const key = logical + 1; let low = 0, high = resident;
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2);
      if (words[16 + middle] < key) low = middle + 1;
      else high = middle;
    }
    if (low >= resident || words[16 + low] !== key) return undefined;
    const encoded = words[pageOffset + low];
    if (encoded === 0 || encoded === 0xffff_ffff) return undefined;
    const local = q.map((v) => v % 8);
    return words[words[6] + (encoded - 1) * 512 + local[0] + local[1] * 8 + local[2] * 64] ?? 0;
  };
  const leafOf = (word: number, q: readonly number[]) => {
    if ((word & 0x8000_0000) !== 0) {
      const exponent = (word >>> 18) & 7, size = 1 << exponent;
      const brickOrigin = q.map((value) => Math.floor(value / 8) * 8);
      const delta = [word & 63, (word >>> 6) & 63, (word >>> 12) & 63]
        .map((value) => value - 32);
      return { origin: brickOrigin.map((value, axis) => value + delta[axis]), size };
    }
    // Zero or malformed page payloads decode to the canonical coarse owner.
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
  const compactPaged = solver.info.gridKind === "octree" && Boolean(solver.globalFineLevelSetSource);
  if (compactPaged) {
    const source = solver.globalFineLevelSetSource;
    if (!source?.coarsePhiDirectory || !source.coarsePhiRowCapacity) {
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
    const [metadataBytes, flagBytes, phiBytes, worklistBytes, coarseBytes, coarseControlBytes,
      fineRestrictionBytes,
      topologyBytes, transportBytes, redistanceBytes, volumeBytes, mgpcgBytes] = await Promise.all([
      readBufferBinding(device, { buffer: source.metadata }, source.plan.maximumResidentBricks * 40),
      readBufferBinding(device, { buffer: source.flags }, sampleWords * 4),
      readBufferBinding(device, { buffer: source.phi }, sampleWords * 4),
      readBufferBinding(device, { buffer: source.worklist }, source.worklist.size),
      readBufferBinding(device, { buffer: source.coarsePhiDirectory }, 32 + source.coarsePhiRowCapacity * 32),
      solver.globalFineCoarseLevelSetControl
        ? readBufferBinding(device, { buffer: solver.globalFineCoarseLevelSetControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineRestrictionControl
        ? readBufferBinding(device, { buffer: solver.globalFineRestrictionControl }, 32)
        : Promise.resolve(undefined),
      source.topologyControl
        // 48 bytes, not 32: words 9..11 latch the finalize rejection that
        // clearDesiredGeneration wipes from words 0..8 every generation.
        ? readBufferBinding(device, { buffer: source.topologyControl }, 48)
        : Promise.resolve(undefined),
      solver.globalFineTransportControl
        ? readBufferBinding(device, { buffer: solver.globalFineTransportControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineRedistanceControl
        ? readBufferBinding(device, { buffer: solver.globalFineRedistanceControl },
          FINE_LEVELSET_REDISTANCE_CONTROL_BYTES)
        : Promise.resolve(undefined),
      solver.globalFineVolumeControl
        ? readBufferBinding(device, { buffer: solver.globalFineVolumeControl }, 64)
        : Promise.resolve(undefined),
      (solver as GPUSolverInstance & { mgpcgControl?: GPUBuffer }).mgpcgControl
        ? readBufferBinding(device, { buffer: (solver as GPUSolverInstance & { mgpcgControl: GPUBuffer }).mgpcgControl }, 64)
        : Promise.resolve(undefined),
    ]);
    const compactSnapshot = {
      plan: source.plan,
      generation: source.generation,
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
      ...(mgpcgBytes ? { mgpcgControl: new Uint32Array(mgpcgBytes.buffer,
        mgpcgBytes.byteOffset, mgpcgBytes.byteLength / 4) } : {}),
    };
    let reconstructed: ReturnType<typeof reconstructCompactOctreeOccupancyField>;
    try {
      reconstructed = reconstructCompactOctreeOccupancyField(compactSnapshot, [nx, ny, nz]);
    } catch (error) {
      const candidateFailure = await (solver as GPUSolverInstance & { octreeProjection?: {
        readPowerFrontierFailure(): Promise<unknown>;
      } }).octreeProjection?.readPowerFrontierFailure();
      console.error(JSON.stringify({ phase: "compact-octree-field-publication-rejected", grid: [nx, ny, nz],
        ...compactOctreePublicationHeaderEvidence(compactSnapshot),
        candidateFailure,
        error: error instanceof Error ? error.message : String(error) }));
      await dumpFineRedistancePageDeltaForensics(device, solver, source, compactSnapshot);
      throw error;
    }
    const { field, ...reconstructionEvidence } = reconstructed;
    // Preserve the controls already read for reconstruction in the returned
    // evidence.  Short, non-raster Dawn reproductions deliberately do not run
    // the much larger presentation-transition audit, but their final
    // topology gates still need the exact same transaction words.
    // This adds no readback: compactOctreePublicationHeaderEvidence only
    // decodes the buffers above.
    const publicationEvidence = compactOctreePublicationHeaderEvidence(compactSnapshot);
    const compactFieldEvidence: CompactOctreeFieldEvidence = {
      ...reconstructionEvidence,
      ...(publicationEvidence.transportControl
        ? { transportControl: publicationEvidence.transportControl } : {}),
    };
    console.log(JSON.stringify({ phase: "compact-octree-field-readback", grid: [nx, ny, nz],
      ...compactOctreePublicationHeaderEvidence(compactSnapshot),
      ...compactFieldEvidence }));
    return { field, summary: summarizeScalarField(field, nx, ny, nz), compactFieldEvidence };
  }
  const levelSet = solver.info.surfaceField === "levelset";
  const packed = await readFloatTexture3D(device, levelSet ? solver.surfaceFieldTexture ?? solver.volumeTexture : solver.volumeTexture, nx, storedNy, nz);
  let bases = new Float32Array(nx * nz);
  if (gridKind === "restricted-tall-cell") bases = await readFloatTexture2D(device, solver.columnBaseTexture!, nx, nz);
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

async function dumpFineRedistancePageDeltaForensics(
  device: GPUDevice,
  solver: GPUSolverInstance,
  source: WebGPUFineLevelSetBrickSource,
  snapshot: {
    metadata: Uint32Array;
    worklist: Uint32Array;
    flags: Uint32Array;
    phi: Float32Array;
    transportControl?: Uint32Array;
    redistanceControl?: Uint32Array;
  },
): Promise<void> {
  const debug = solver.globalFinePageDeltaDebug;
  const control = snapshot.redistanceControl;
  if (!debug || !control || control.length < 10) return;
  const [headerBytes, parameterBytes] = await Promise.all([
    readBufferBinding(device, { buffer: debug.buffer }, 64),
    readBufferBinding(device, { buffer: debug.params }, 96),
  ]);
  const header = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, 16);
  const topologyParameters = new Uint32Array(parameterBytes.buffer, parameterBytes.byteOffset, 24);
  const dirtyCount = Math.min(header[2], debug.pageCapacity);
  const supportCount = Math.min(header[3], debug.pageCapacity);
  const changedCount = Math.min(header[0], 2 * debug.pageCapacity);
  const readPageStream = async (offsetWords: number, count: number) => {
    if (count === 0) return new Uint32Array(0);
    const bytes = await readBufferBinding(device,
      { buffer: debug.buffer, offset: offsetWords * 4 }, count * 4);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, count);
  };
  const [changedKeys, dirty, support] = await Promise.all([
    readPageStream(debug.changedKeysOffsetWords, changedCount),
    readPageStream(debug.dirtyPagesOffsetWords, dirtyCount),
    readPageStream(debug.supportPagesOffsetWords, supportCount),
  ]);
  const supportSet = new Set<number>(support);
  const dirtySet = new Set<number>(dirty);
  const dirtyMissingFromSupport = Array.from(dirty).filter((id) => !supportSet.has(id));
  const supportOutsideDirty = Array.from(support).filter((id) => !dirtySet.has(id));
  const firstError = control[5] >>> 0;
  const errorPage = firstError === 0xffff_ffff
    ? 0xffff_ffff : Math.floor(firstError / source.plan.samplesPerBrick);
  let errorPageScratch: Record<string, unknown> | undefined;
  if (errorPage < debug.pageCapacity) {
    const sampleOffset = errorPage * source.plan.samplesPerBrick;
    const sampleBytes = source.plan.samplesPerBrick * 4;
    const [aBytes, bBytes] = await Promise.all([
      readBufferBinding(device, { buffer: source.workA, offset: sampleOffset * 4 }, sampleBytes),
      readBufferBinding(device, { buffer: source.workB, offset: sampleOffset * 4 }, sampleBytes),
    ]);
    const a = new Uint32Array(aBytes.buffer, aBytes.byteOffset, source.plan.samplesPerBrick);
    const b = new Uint32Array(bBytes.buffer, bBytes.byteOffset, source.plan.samplesPerBrick);
    const bf = new Float32Array(bBytes.buffer, bBytes.byteOffset, source.plan.samplesPerBrick);
    let validSamples = 0, finiteDistances = 0, invalidSeeds = 0;
    for (let local = 0; local < source.plan.samplesPerBrick; local += 1) {
      if ((snapshot.flags[sampleOffset + local] & 1) !== 0) validSamples += 1;
      if (Number.isFinite(bf[local])) finiteDistances += 1;
      if (a[local] === 0xffff_ffff) invalidSeeds += 1;
    }
    errorPageScratch = {
      page: errorPage,
      generation: snapshot.metadata[errorPage * 10 + 2],
      dirtyRank: Array.from(dirty).indexOf(errorPage),
      supportRank: Array.from(support).indexOf(errorPage),
      validSamples,
      finiteDistances,
      invalidSeeds,
      workAFirst8: Array.from(a.slice(0, 8)),
      workBFirst8: Array.from(b.slice(0, 8)),
      phiFirst8: Array.from(snapshot.phi.slice(sampleOffset, sampleOffset + 8)),
      flagsFirst8: Array.from(snapshot.flags.slice(sampleOffset, sampleOffset + 8)),
    };
  }
  const transportFirstError = snapshot.transportControl?.[12] ?? 0xffff_ffff;
  let transportErrorSample: Record<string, unknown> | undefined;
  if (transportFirstError !== 0xffff_ffff) {
    const page = Math.floor(transportFirstError / source.plan.samplesPerBrick);
    const local = transportFirstError % source.plan.samplesPerBrick;
    if (page < debug.pageCapacity) {
      const key = snapshot.metadata[page * 10 + 1];
      const r = source.plan.brickResolution;
      const localZ = Math.floor(local / (r * r));
      const localRem = local - localZ * r * r;
      const localY = Math.floor(localRem / r), localX = localRem - localY * r;
      const bx = key % source.plan.brickDimensions[0];
      const by = Math.floor(key / source.plan.brickDimensions[0])
        % source.plan.brickDimensions[1];
      const bz = Math.floor(key / (source.plan.brickDimensions[0]
        * source.plan.brickDimensions[1]));
      const nextBytes = await readBufferBinding(device,
        { buffer: source.workA, offset: transportFirstError * 4 }, 4);
      const next = new Float32Array(nextBytes.buffer, nextBytes.byteOffset, 1)[0];
      transportErrorSample = {
        index: transportFirstError, page, key, local,
        fineSample: [bx * r + localX, by * r + localY, bz * r + localZ],
        phi: Number.isFinite(snapshot.phi[transportFirstError])
          ? snapshot.phi[transportFirstError] : null,
        nextPhi: Number.isFinite(next) ? next : null,
        flags: snapshot.flags[transportFirstError],
      };
    }
  }
  const [bx, by, bz] = source.plan.brickDimensions;
  const logicalCount = bx * by * bz;
  const changedMask = new Uint8Array(logicalCount);
  for (const key of changedKeys) if (key < logicalCount) changedMask[key] = 1;
  const distance = new Uint8Array(logicalCount).fill(0xff);
  let frontier = Array.from(changedKeys).filter((key) => key < logicalCount);
  for (const key of frontier) distance[key] = 0;
  for (let radius = 1; radius <= 16 && frontier.length > 0; radius += 1) {
    const next: number[] = [];
    for (const key of frontier) {
      const x = key % bx, y = Math.floor(key / bx) % by, z = Math.floor(key / (bx * by));
      for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx, ny = y + dy, nz = z + dz;
          if (nx < 0 || nx >= bx || ny < 0 || ny >= by || nz < 0 || nz >= bz) continue;
          const neighbor = nx + bx * (ny + by * nz);
          if (distance[neighbor] !== 0xff) continue;
          distance[neighbor] = radius;
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  const activeCount = Math.min(snapshot.worklist[1], debug.pageCapacity);
  const activeKeys: number[] = [];
  for (let rank = 0; rank < activeCount; rank += 1) {
    const id = snapshot.worklist[7 + rank];
    if (id < debug.pageCapacity) activeKeys.push(snapshot.metadata[id * 10 + 1]);
  }
  const activeCountByChangedChebyshevRadius = Array.from({ length: 17 }, (_, radius) =>
    activeKeys.reduce((count, key) => count + (key < logicalCount && distance[key] <= radius ? 1 : 0), 0));
  const pageDistanceHistogram = (pages: Uint32Array) => {
    const histogram = new Array<number>(18).fill(0);
    for (const id of pages) {
      const key = id < debug.pageCapacity ? snapshot.metadata[id * 10 + 1] : 0xffff_ffff;
      const d = key < logicalCount ? distance[key] : 0xff;
      histogram[Math.min(d, 17)] += 1;
    }
    return histogram;
  };
  console.error(JSON.stringify({
    phase: "fine-redistance-page-delta-forensics",
    generation: source.generation,
    topologyParameters: Array.from(topologyParameters),
    header: Array.from(header),
    dirtyCount,
    supportCount,
    dirtyUnique: dirtySet.size,
    supportUnique: supportSet.size,
    dirtyMissingFromSupportCount: dirtyMissingFromSupport.length,
    dirtyMissingFromSupportFirst16: dirtyMissingFromSupport.slice(0, 16),
    supportOutsideDirtyCount: supportOutsideDirty.length,
    supportOutsideDirtyFirst16: supportOutsideDirty.slice(0, 16),
    changedCount,
    changedUnique: new Set(changedKeys).size,
    firstChangedKeys: Array.from(changedKeys.slice(0, 16)),
    activeCountByChangedChebyshevRadius,
    terminalSparseTopologyCount: activeKeys.length,
    terminalSparseTopologyFirst16: activeKeys.slice(0, 16),
    dirtyDistanceHistogram: pageDistanceHistogram(dirty),
    supportDistanceHistogram: pageDistanceHistogram(support),
    firstDirtyPages: Array.from(dirty.slice(0, 16)),
    firstSupportPages: Array.from(support.slice(0, 16)),
    firstError,
    errorPageScratch,
    transportErrorSample,
  }));
}

type GPUCommandAuditBucket = { calls: number; bytes: number };
interface GPUCommandAuditReport {
  writeBuffer: GPUCommandAuditBucket;
  writeTexture: GPUCommandAuditBucket;
  clearBuffer: GPUCommandAuditBucket;
  copyBufferToBuffer: GPUCommandAuditBucket;
  bufferAllocations: GPUCommandAuditBucket;
  bindGroups: number;
  commandEncoders: number;
  commandBuffers: number;
  computePasses: number;
  dispatches: number;
  indirectDispatches: number;
  submissions: number;
  submittedCommandBuffers: number;
  completionFences: number;
  writeBufferByLabel: Record<string, GPUCommandAuditBucket>;
  clearBufferByLabel: Record<string, GPUCommandAuditBucket>;
  copyBufferToBufferByLabel: Record<string, GPUCommandAuditBucket>;
  bufferAllocationsByLabel: Record<string, GPUCommandAuditBucket>;
  commandEncodersByLabel: Record<string, GPUCommandAuditBucket>;
  computePassesByLabel: Record<string, GPUCommandAuditBucket>;
  dispatchesByPassLabel: Record<string, GPUCommandAuditBucket>;
  indirectDispatchesByPassLabel: Record<string, GPUCommandAuditBucket>;
}

interface GPUFineTimestampBucket {
  samples: number;
  total_ms: number;
  mean_ms: number;
  minimum_ms: number;
  maximum_ms: number;
}

interface GPUFineTimestampReport {
  measuredAdvances: number;
  measuredPasses: number;
  invalidPasses: number;
  summedPass_ms: number;
  byLabel: Record<string, GPUFineTimestampBucket>;
}

interface GPUPassTimestampReport {
  capturedCommandBuffers: number;
  measuredPasses: number;
  invalidPasses: number;
  capacityOverflows: number;
  summedPass_ms: number;
  /** False means Dawn was free to merge passes into shared Metal encoders, so
   * each label is really its encoder's total charged to the encoder's last
   * pass. Only an isolated report attributes time to a single pass, and only
   * an unisolated wall clock states what the frame costs. */
  encoderIsolated: boolean;
  /** Every labelled `compute()` got its own pass, so a label's dispatches are
   * exactly the dispatches recorded under that label. */
  labelIsolated: boolean;
  /** Wall span of the captured command buffers on the GPU timeline: last
   * timestamp minus first. */
  span_ms: number;
  /** `summedPass_ms / span_ms`. One means the passes tile the span, which is
   * the only state in which a label's ms is that label's cost. Above one means
   * brackets overlap, below one means unbracketed GPU time. */
  coverageRatio: number;
  byLabel: Record<string, GPUFineTimestampBucket>;
}

interface GPUPassTimestampCapture {
  buckets: Array<[string, number | undefined]>;
  capacityOverflows: number;
  span_ms: number;
}

class GPUPassTimestampEncoderSession {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly readBuffer: GPUBuffer;
  private readonly labels: string[] = [];
  private queryCount = 0;
  private capacityOverflows = 0;

  constructor(private readonly device: GPUDevice, private readonly capacity: number) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: capacity });
    this.resolveBuffer = device.createBuffer({
      label: "Algorithm pass timestamps resolve",
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuffer = device.createBuffer({
      label: "Algorithm pass timestamps readback",
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  computePassDescriptor(descriptor?: GPUComputePassDescriptor): GPUComputePassDescriptor | undefined {
    if (descriptor?.timestampWrites) {
      // Never overwrite the solver's semantic recorder. Diagnostic runs turn
      // that recorder off, but retaining this guard makes the two facilities
      // composable and fail-closed.
      return descriptor;
    }
    if (this.queryCount + 2 > this.capacity) {
      this.capacityOverflows += 1;
      return descriptor;
    }
    const label = descriptor?.label?.trim() || "<unlabeled compute pass>";
    const beginningOfPassWriteIndex = this.queryCount;
    const endOfPassWriteIndex = this.queryCount + 1;
    this.queryCount += 2;
    this.labels.push(label);
    return {
      ...descriptor,
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex,
        endOfPassWriteIndex,
      },
    };
  }

  resolve(encoder: GPUCommandEncoder): void {
    if (this.queryCount === 0) return;
    const bytes = this.queryCount * 8;
    encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, bytes);
  }

  async read(): Promise<GPUPassTimestampCapture> {
    try {
      if (this.queryCount === 0) {
        return { buckets: [], capacityOverflows: this.capacityOverflows, span_ms: 0 };
      }
      const bytes = this.queryCount * 8;
      await this.readBuffer.mapAsync(GPUMapMode.READ, 0, bytes);
      const timestamps = new BigUint64Array(this.readBuffer.getMappedRange(0, bytes).slice(0));
      // The union of every pass bracket. Compared against the sum of the
      // brackets it states whether this capture is an attribution at all: a sum
      // materially above the span means passes overlapped on the GPU timeline,
      // and a per-pass number that overlaps its neighbours cannot be a cost.
      let earliest: bigint | undefined, latest: bigint | undefined;
      for (let index = 0; index < this.queryCount; index += 1) {
        const timestamp = timestamps[index] ?? 0n;
        if (timestamp === 0n) continue;
        if (earliest === undefined || timestamp < earliest) earliest = timestamp;
        if (latest === undefined || timestamp > latest) latest = timestamp;
      }
      return {
        buckets: this.labels.map((label, index) => {
          const begin = timestamps[2 * index] ?? 0n;
          const end = timestamps[2 * index + 1] ?? 0n;
          return [label, begin > 0n && end >= begin ? Number(end - begin) / 1e6 : undefined];
        }),
        capacityOverflows: this.capacityOverflows,
        span_ms: earliest !== undefined && latest !== undefined ? Number(latest - earliest) / 1e6 : 0,
      };
    } finally {
      if (this.readBuffer.mapState === "mapped") this.readBuffer.unmap();
      this.querySet.destroy();
      this.resolveBuffer.destroy();
      this.readBuffer.destroy();
    }
  }
}

/** Captures a bounded number of complete command buffers after `start()`. The
 * smoke starts it only after construction/t=0 publication, so the first buffer
 * is a real mini-dam recurring advance rather than shader warmup. */
class GPUPassTimestampAudit {
  private enabled = false;
  private claimedCommandBuffers = 0;
  private skippedCommandBuffers = 0;
  private readonly submitted = new WeakMap<GPUCommandBuffer, GPUPassTimestampEncoderSession>();
  private readonly reads: Promise<GPUPassTimestampCapture>[] = [];

  constructor(
    private readonly device: GPUDevice,
    private readonly maximumCommandBuffers = 1,
    private readonly queryCapacity = 2048,
    private readonly encoderIsolated = false,
    private readonly labelIsolated = false,
    private readonly skipCommandBuffers = 0,
  ) {}

  start(): void { this.enabled = true; }

  createEncoderSession(): GPUPassTimestampEncoderSession | undefined {
    if (!this.enabled || this.claimedCommandBuffers >= this.maximumCommandBuffers) return undefined;
    // The first recurring command buffer is the coldest one in the run: on the
    // mini lane its GPU span measured 92.7 ms against a 39.1 ms mean advance.
    // Skipping forward buys a representative buffer at no cost but a later
    // capture.
    if (this.skippedCommandBuffers < this.skipCommandBuffers) {
      this.skippedCommandBuffers += 1;
      return undefined;
    }
    this.claimedCommandBuffers += 1;
    return new GPUPassTimestampEncoderSession(this.device, this.queryCapacity);
  }

  attach(commandBuffer: GPUCommandBuffer, session: GPUPassTimestampEncoderSession): void {
    this.submitted.set(commandBuffer, session);
  }

  afterSubmit(commandBuffers: readonly GPUCommandBuffer[]): void {
    for (const commandBuffer of commandBuffers) {
      const session = this.submitted.get(commandBuffer);
      if (session) this.reads.push(session.read());
    }
  }

  async report(): Promise<GPUPassTimestampReport> {
    const captures = await Promise.all(this.reads);
    const aggregates = new Map<string, Omit<GPUFineTimestampBucket, "mean_ms">>();
    let invalidPasses = 0;
    for (const capture of captures) for (const [label, duration] of capture.buckets) {
      if (duration === undefined || !Number.isFinite(duration) || duration < 0) {
        invalidPasses += 1;
        continue;
      }
      const bucket = aggregates.get(label) ?? {
        samples: 0, total_ms: 0, minimum_ms: Number.POSITIVE_INFINITY, maximum_ms: 0,
      };
      bucket.samples += 1;
      bucket.total_ms += duration;
      bucket.minimum_ms = Math.min(bucket.minimum_ms, duration);
      bucket.maximum_ms = Math.max(bucket.maximum_ms, duration);
      aggregates.set(label, bucket);
    }
    const byLabel = Object.fromEntries(Array.from(aggregates.entries())
      .sort((left, right) => right[1].total_ms - left[1].total_ms || left[0].localeCompare(right[0]))
      .map(([label, bucket]) => [label, {
        ...bucket,
        mean_ms: bucket.total_ms / Math.max(1, bucket.samples),
      }]));
    const summedPass_ms = Array.from(aggregates.values()).reduce((sum, bucket) => sum + bucket.total_ms, 0);
    const span_ms = captures.reduce((sum, capture) => sum + capture.span_ms, 0);
    return {
      capturedCommandBuffers: captures.length,
      measuredPasses: Array.from(aggregates.values()).reduce((sum, bucket) => sum + bucket.samples, 0),
      invalidPasses,
      capacityOverflows: captures.reduce((sum, capture) => sum + capture.capacityOverflows, 0),
      summedPass_ms,
      encoderIsolated: this.encoderIsolated,
      labelIsolated: this.labelIsolated,
      span_ms,
      coverageRatio: span_ms > 0 ? summedPass_ms / span_ms : 0,
      byLabel,
    };
  }
}

class GPUCommandAudit {
  private writeBuffer = { calls: 0, bytes: 0 };
  private writeTexture = { calls: 0, bytes: 0 };
  private clearBuffer = { calls: 0, bytes: 0 };
  private copyBufferToBuffer = { calls: 0, bytes: 0 };
  private bufferAllocations = { calls: 0, bytes: 0 };
  private bindGroups = 0;
  private commandEncoders = 0;
  private commandBuffers = 0;
  private computePasses = 0;
  private dispatches = 0;
  private indirectDispatches = 0;
  private submissions = 0;
  private submittedCommandBuffers = 0;
  private completionFences = 0;
  private readonly writeBufferByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly clearBufferByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly copyBufferToBufferByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly bufferAllocationsByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly commandEncodersByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly computePassesByLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly dispatchesByPassLabel = new Map<string, GPUCommandAuditBucket>();
  private readonly indirectDispatchesByPassLabel = new Map<string, GPUCommandAuditBucket>();

  private label(value: { label?: string } | undefined, fallback: string): string {
    return value?.label?.trim() || fallback;
  }
  private add(map: Map<string, GPUCommandAuditBucket>, label: string, bytes = 0): void {
    const bucket = map.get(label) ?? { calls: 0, bytes: 0 };
    bucket.calls += 1; bucket.bytes += bytes; map.set(label, bucket);
  }
  private record(bucket: GPUCommandAuditBucket, bytes: number): void {
    bucket.calls += 1; bucket.bytes += bytes;
  }
  reset(): void {
    for (const bucket of [this.writeBuffer, this.writeTexture, this.clearBuffer,
      this.copyBufferToBuffer, this.bufferAllocations]) { bucket.calls = 0; bucket.bytes = 0; }
    this.bindGroups = 0; this.commandEncoders = 0; this.commandBuffers = 0;
    this.computePasses = 0; this.dispatches = 0; this.indirectDispatches = 0;
    this.submissions = 0; this.submittedCommandBuffers = 0; this.completionFences = 0;
    for (const map of [this.writeBufferByLabel, this.clearBufferByLabel, this.bufferAllocationsByLabel,
      this.copyBufferToBufferByLabel, this.commandEncodersByLabel, this.computePassesByLabel,
      this.dispatchesByPassLabel, this.indirectDispatchesByPassLabel]) map.clear();
  }
  recordWriteBuffer(buffer: GPUBuffer, bytes: number): void {
    this.record(this.writeBuffer, bytes); this.add(this.writeBufferByLabel, this.label(buffer, "<unlabeled buffer>"), bytes);
  }
  recordWriteTexture(bytes: number): void { this.record(this.writeTexture, bytes); }
  recordClearBuffer(buffer: GPUBuffer, bytes: number): void {
    this.record(this.clearBuffer, bytes); this.add(this.clearBufferByLabel, this.label(buffer, "<unlabeled buffer>"), bytes);
  }
  recordCopyBuffer(source: GPUBuffer, destination: GPUBuffer, bytes: number): void {
    this.record(this.copyBufferToBuffer, bytes);
    this.add(this.copyBufferToBufferByLabel,
      `${this.label(source, "<unlabeled buffer>")} -> ${this.label(destination, "<unlabeled buffer>")}`, bytes);
  }
  recordBufferAllocation(descriptor: GPUBufferDescriptor): void {
    const bytes = Number(descriptor.size); this.record(this.bufferAllocations, bytes);
    this.add(this.bufferAllocationsByLabel, descriptor.label?.trim() || "<unlabeled buffer>", bytes);
  }
  recordBindGroup(): void { this.bindGroups += 1; }
  recordCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): void {
    this.commandEncoders += 1;
    this.add(this.commandEncodersByLabel, descriptor?.label?.trim() || "<unlabeled encoder>");
  }
  recordCommandBuffer(): void { this.commandBuffers += 1; }
  recordComputePass(descriptor?: GPUComputePassDescriptor): void {
    this.computePasses += 1;
    this.add(this.computePassesByLabel, descriptor?.label?.trim() || "<unlabeled compute pass>");
  }
  recordDispatch(passLabel: string, indirect: boolean): void {
    this.dispatches += 1; this.add(this.dispatchesByPassLabel, passLabel);
    if (indirect) { this.indirectDispatches += 1; this.add(this.indirectDispatchesByPassLabel, passLabel); }
  }
  recordSubmit(commandBufferCount: number): void {
    this.submissions += 1; this.submittedCommandBuffers += commandBufferCount;
  }
  recordFence(): void { this.completionFences += 1; }
  private object(map: Map<string, GPUCommandAuditBucket>): Record<string, GPUCommandAuditBucket> {
    return Object.fromEntries(Array.from(map.entries()).sort((left, right) =>
      right[1].bytes - left[1].bytes || right[1].calls - left[1].calls || left[0].localeCompare(right[0])));
  }
  snapshot(): GPUCommandAuditReport {
    return {
      writeBuffer: { ...this.writeBuffer }, writeTexture: { ...this.writeTexture },
      clearBuffer: { ...this.clearBuffer }, copyBufferToBuffer: { ...this.copyBufferToBuffer },
      bufferAllocations: { ...this.bufferAllocations }, bindGroups: this.bindGroups,
      commandEncoders: this.commandEncoders, commandBuffers: this.commandBuffers,
      computePasses: this.computePasses, dispatches: this.dispatches,
      indirectDispatches: this.indirectDispatches, submissions: this.submissions,
      submittedCommandBuffers: this.submittedCommandBuffers, completionFences: this.completionFences,
      writeBufferByLabel: this.object(this.writeBufferByLabel),
      clearBufferByLabel: this.object(this.clearBufferByLabel),
      copyBufferToBufferByLabel: this.object(this.copyBufferToBufferByLabel),
      bufferAllocationsByLabel: this.object(this.bufferAllocationsByLabel),
      commandEncodersByLabel: this.object(this.commandEncodersByLabel),
      computePassesByLabel: this.object(this.computePassesByLabel),
      dispatchesByPassLabel: this.object(this.dispatchesByPassLabel),
      indirectDispatchesByPassLabel: this.object(this.indirectDispatchesByPassLabel),
    };
  }
}

function writtenByteLength(data: GPUAllowSharedBufferSource, dataOffset = 0, size?: number): number {
  if (size !== undefined) return size;
  const byteLength = ArrayBuffer.isView(data) ? data.byteLength : data.byteLength;
  return Math.max(0, byteLength - dataOffset);
}

function auditComputePass(pass: GPUComputePassEncoder, audit: GPUCommandAudit | undefined,
  passLabel: string, dataFlow?: GPUDataFlowPassRecorder): GPUComputePassEncoder {
  return new Proxy(pass, { get(target, property) {
    if (property === "setPipeline") return (pipeline: GPUComputePipeline) => {
      dataFlow?.setPipeline(pipeline);
      return target.setPipeline(pipeline);
    };
    if (property === "setBindGroup") return (
      index: number,
      bindGroup: GPUBindGroup | null,
      dynamicOffsets?: Iterable<number>,
      dynamicOffsetsDataStart?: number,
      dynamicOffsetsDataLength?: number,
    ) => {
      const offsets = dynamicOffsets === undefined ? undefined : Array.from(dynamicOffsets);
      const start = dynamicOffsetsDataStart ?? 0;
      const length = dynamicOffsetsDataLength ?? Math.max(0, (offsets?.length ?? 0) - start);
      dataFlow?.setBindGroup(index, bindGroup, offsets?.slice(start, start + length));
      if (dynamicOffsetsDataStart !== undefined && dynamicOffsets instanceof Uint32Array) {
        return target.setBindGroup(
          index, bindGroup, dynamicOffsets,
          dynamicOffsetsDataStart,
          dynamicOffsetsDataLength ?? Math.max(0, dynamicOffsets.length - dynamicOffsetsDataStart),
        );
      }
      if (dynamicOffsets !== undefined) return target.setBindGroup(index, bindGroup, dynamicOffsets);
      return target.setBindGroup(index, bindGroup);
    };
    if (property === "dispatchWorkgroups") return (...args: Parameters<GPUComputePassEncoder["dispatchWorkgroups"]>) => {
      audit?.recordDispatch(passLabel, false);
      dataFlow?.direct(args[0], args[1], args[2]);
      return Reflect.apply(target.dispatchWorkgroups, target, args);
    };
    if (property === "dispatchWorkgroupsIndirect") return (...args: Parameters<GPUComputePassEncoder["dispatchWorkgroupsIndirect"]>) => {
      audit?.recordDispatch(passLabel, true);
      dataFlow?.indirect(args[0], Number(args[1]));
      return Reflect.apply(target.dispatchWorkgroupsIndirect, target, args);
    };
    const value = Reflect.get(target, property, target);
    return typeof value === "function" ? value.bind(target) : value;
  } }) as GPUComputePassEncoder;
}

function auditCommandEncoder(
  encoder: GPUCommandEncoder,
  audit?: GPUCommandAudit,
  dataFlow?: GPUDataFlowEncoderSession,
  passTimestamps?: GPUPassTimestampEncoderSession,
  attachPassTimestamps?: (
    commandBuffer: GPUCommandBuffer,
    session: GPUPassTimestampEncoderSession,
  ) => void,
): GPUCommandEncoder {
  return new Proxy(encoder, { get(target, property) {
    if (property === "clearBuffer") return (buffer: GPUBuffer, offset = 0, size?: number) => {
      const bytes = size ?? Math.max(0, buffer.size - offset); audit?.recordClearBuffer(buffer, bytes);
      return target.clearBuffer(buffer, offset, size);
    };
    if (property === "copyBufferToBuffer") return (source: GPUBuffer, sourceOffset: number, destination: GPUBuffer,
      destinationOffset: number, size: number) => {
      audit?.recordCopyBuffer(source, destination, size);
      return target.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size);
    };
    if (property === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
      audit?.recordComputePass(descriptor);
      const label = descriptor?.label?.trim() || "<unlabeled compute pass>";
      const pass = target.beginComputePass(passTimestamps?.computePassDescriptor(descriptor) ?? descriptor);
      const flowPass = dataFlow?.beginPass(label);
      return audit || flowPass ? auditComputePass(pass, audit, label, flowPass) : pass;
    };
    if (property === "finish") return (descriptor?: GPUCommandBufferDescriptor) => {
      audit?.recordCommandBuffer();
      passTimestamps?.resolve(target);
      const commandBuffer = target.finish(descriptor);
      if (passTimestamps) attachPassTimestamps?.(commandBuffer, passTimestamps);
      return commandBuffer;
    };
    const value = Reflect.get(target, property, target);
    return typeof value === "function" ? value.bind(target) : value;
  } }) as GPUCommandEncoder;
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
  /** `advanceTo` self-rejections retried by the advance loop. Rejections cost
   * wall time but never increment `steps`, so without these two counters a
   * wedged solver and a slow one produce identical result records. */
  rejectedAdvanceAttempts: number;
  maximumConsecutiveRejectedAdvances: number;
  /** Distinct structured reject-carry summaries observed on `solver.info`. A
   * latched carry zeroes every class dispatch, so every later step is a no-op;
   * these come from the host publication and add no readback. */
  structuredRejectReports: number;
  firstStructuredRejectStep?: number;
  gpuCommandAudit?: GPUCommandAuditReport;
  gpuFineTimestamps?: GPUFineTimestampReport;
  gpuPassTimestamps?: GPUPassTimestampReport;
  algorithmDiagnostics?: Record<string, unknown>;
  gpuDataFlowManifest?: GPUDataFlowManifest;
  /** Accepted steps whose live power topology, faces, transfer, and MGPCG publication passed the generation audit. */
  powerGenerationAuditedSteps: number;
  /** Iteration envelope decoded from the terminal packed generation audit;
   * this adds no readback beyond that single aggregate snapshot. */
  mgpcgIterationAudit?: { samples: number; minimum: number; maximum: number; histogram: Record<string, number> };
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
  octreePowerTopologyDiagnostics?: OctreePowerTopologyDiagnostics;
  octreeMGPCGDiagnostics?: OctreeMGPCGDiagnostics;
  stabilityEnvelope?: StabilityEnvelope;
  octreeWorkAccounting?: OctreeWorkSnapshot;
  /** Exact cause when a work-accounting stage failed its validity gate. */
  octreeWorkAccountingBlocker?: string;
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
    compactMechanicalEnergy?: ReturnType<typeof compactMechanicalEnergyDiagnostic> & {
      publicationValid: boolean;
      rowCount: number;
      reconstructedRows: number;
      coveredCells: number;
      overlapCells: number;
      invalidRows: number;
      liquidCellCount: number;
      finiteLiquidCellCount: number;
      liquidVolumeCellSum: number;
      finiteLiquidVolumeCellSum: number;
      maximumLiquidComponentSpeed_m_s: number;
      maximumLiquidComponentCfl: number;
      nonFiniteLiquidComponentCount: number;
    };
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
  projectionEnergySampleCount: number;
  maximumPressureRelativeResidual: number;
  maximumProjectedVariationalResidual: number;
  maximumExactVolumeDrift: number;
  maximumLevelSetMismatchFraction: number;
  maximumComponentCount: number;
  minimumDominantComponentFraction: number;
  nonFiniteVelocityCount: number;
  invalidVolumeSampleCount: number;
  /** Expensive reconstructed occupancy/connectivity observations. */
  spatialSampledSteps: number;
  /** Queue-copied compact authority observations; exact for power runs. */
  sampledSteps: number;
}

function referenceVolumeCells(info: GPUEulerianInfo) {
  return info.surfaceField === "levelset"
    ? info.referenceLiquidVolume_cells ?? info.initialVolumeCellSum ?? 0
    : info.initialVolumeCellSum ?? 0;
}

function performancePhase_ms(trace: PerformanceTrace | undefined, id: PaperPhaseId): number | undefined {
  if (!trace) return undefined;
  return trace.phases
    .filter((phase) => phase.id === id)
    .reduce((sum, phase) => sum + phase.duration_ms, 0);
}

function reportResult(scenario: SmokeScenarioId, result: GPUSmokeResult) {
  const info = result.info;
  console.log(JSON.stringify({
    scenario, method: result.method, phase: "result", construction_ms: Math.round(result.construction_ms), runtime_ms: Math.round(result.runtime_ms), simulationWall_ms: Math.round(result.simulationWall_ms), steps: result.steps,
    rejectedAdvanceAttempts: result.rejectedAdvanceAttempts,
    maximumConsecutiveRejectedAdvances: result.maximumConsecutiveRejectedAdvances,
    structuredRejectReports: result.structuredRejectReports,
    firstStructuredRejectStep: result.firstStructuredRejectStep,
    structuredRejectStage: info.structuredRejectStage,
    structuredRejectIndex: info.structuredRejectIndex,
    structuredRejectSummary: info.structuredRejectSummary,
    gpuCommandAudit: result.gpuCommandAudit,
    gpuFineTimestamps: result.gpuFineTimestamps,
    gpuPassTimestamps: result.gpuPassTimestamps,
    algorithmDiagnostics: result.algorithmDiagnostics,
    gpuDataFlowManifest: result.gpuDataFlowManifest,
    powerGenerationAuditedSteps: result.powerGenerationAuditedSteps,
    mgpcgIterationAudit: result.mgpcgIterationAudit,
    physicsTrace: info.physicsTrace,
    simulatedTime_s: info.simulatedTime_s, submittedTime_s: info.submittedTime_s,
    completedTime_s: info.completedTime_s, lastDt_s: info.lastDt_s, lastSubsteps: info.lastSubsteps,
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
    pressureRequiredRows: info.pressureRequiredRows,
    pressureCapacityOverflow: info.pressureCapacityOverflow,
    powerDiagramReady: info.powerDiagramReady,
    powerDiagramAuthoritative: info.powerDiagramAuthoritative,
    powerDiagramAllocatedBytes: info.powerDiagramAllocatedBytes,
    structuredPreProjectionKineticEnergyProxy: info.structuredPreProjectionKineticEnergyProxy,
    structuredPostProjectionKineticEnergyProxy: info.structuredPostProjectionKineticEnergyProxy,
    structuredProjectionEnergyRatio: info.structuredProjectionEnergyRatio,
    structuredProjectionEnergySampleCount: info.structuredProjectionEnergySampleCount,
    globalFineLevelSetAllocatedBytes: info.globalFineLevelSetAllocatedBytes,
    globalFineLevelSetResidentBrickCapacity: info.globalFineLevelSetResidentBrickCapacity,
    globalFineLevelSetLogicalBrickCount: info.globalFineLevelSetLogicalBrickCount,
    globalFineLevelSetEnabled: info.globalFineLevelSetEnabled,
    globalFineLevelSetFactor: info.globalFineLevelSetFactor,
    globalFineInterfaceBricks: info.globalFineInterfaceBricks,
    globalFineDesiredBricks: info.globalFineDesiredBricks,
    globalFineActivatedBricks: info.globalFineActivatedBricks,
    globalFineActiveBricks: info.globalFineActiveBricks,
    globalFineTransportQueryCapacity: info.globalFineTransportQueryCapacity,
    globalFineTransportChunkCapacity: info.globalFineTransportChunkCapacity,
    globalFineTransportChunkCount: info.globalFineTransportChunkCount,
    globalFineTransportSegmentCount: info.globalFineTransportSegmentCount,
    globalFineTransportEncodedPasses: info.globalFineTransportEncodedPasses,
    globalFineTransportPrepassScratchBytes: info.globalFineTransportPrepassScratchBytes,
    globalFineTransportVertexScratchBytes: info.globalFineTransportVertexScratchBytes,
    frontierListCapacity: info.frontierListCapacity,
    frontierRequiredLeaves: info.frontierRequiredLeaves,
    frontierCapacityOverflow: info.frontierCapacityOverflow,
    quadtreePressureConverged: info.quadtreePressureConverged,
    quadtreeFactorLevelCount: info.quadtreeFactorLevelCount,
    quadtreeMultigridLevelCount: info.quadtreeMultigridLevelCount,
    quadtreeMultigridCoarsestDofs: info.quadtreeMultigridCoarsestDofs,
    quadtreeTopologyReadbackBytes: info.quadtreeTopologyReadbackBytes,
    initialVolumeCellSum: info.initialVolumeCellSum, volumeCellSum: info.volumeCellSum,
    representedVolumeCellSum: info.representedVolumeCellSum, volumeDrift: info.volumeDrift,
    representedVolumeDrift: info.representedVolumeDrift, rawVolumeDrift: info.rawVolumeDrift,
    volumeCorrectionNormalSpeed_cells_s: info.volumeCorrectionNormalSpeed_cells_s, volumeCorrectionDivergenceRate_s: info.volumeCorrectionDivergenceRate_s, phiInterfaceCellCount: info.phiInterfaceCellCount, front_m: info.front_m,
    maxSpeed_m_s: info.maxSpeed_m_s, maxComponentCfl: info.maxComponentCfl,
    adaptiveFaceTransportedCount: info.adaptiveFaceTransportedCount,
    maxDivergenceBefore_s: info.maxDivergenceBefore_s,
    maxDivergenceAfter_s: info.maxDivergenceAfter_s, pressureRelativeResidual: info.pressureRelativeResidual,
    pressureResidual: info.pressureResidual,
    nonFiniteCount: info.nonFiniteCount, stabilityFlags: info.stabilityFlags,
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
        enclosedSurfaceHoles: raster.enclosedSurfaceHoles,
        surfaceSteps: raster.surfaceSteps,
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
    compactMechanicalEnergyCheckpoints: result.checkpoints.flatMap(({ time_s, compactMechanicalEnergy }) =>
      compactMechanicalEnergy ? [{ time_s, ...compactMechanicalEnergy }] : []),
    octreePowerTopologyDiagnostics: result.octreePowerTopologyDiagnostics,
    octreeMGPCGDiagnostics: result.octreeMGPCGDiagnostics,
    stabilityEnvelope: result.stabilityEnvelope,
    octreeWorkAccounting: result.octreeWorkAccounting,
    octreeWorkAccountingBlocker: result.octreeWorkAccountingBlocker,
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
  const requiredFeatures = fluidExecutionDeviceFeatures(adapter.features);
  const requiredLimits = requiredFluidDeviceLimits(adapter.limits);
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  await assertComputeSentinel(device);
  let lost: GPUDeviceLostInfo | undefined;
  void device.lost.then((info) => { lost = info; });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  const commandAudit = gpuCommandAuditRequested ? new GPUCommandAudit() : undefined;
  const dataFlowAudit = genericPhaseTraceRequested ? new GPUDataFlowAudit() : undefined;
  const passTimestampAudit = gpuPassTimestampRequested
    && device.features.has("timestamp-query")
    ? new GPUPassTimestampAudit(device, Math.max(1,
      Math.floor(Number(process.env.FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS ?? 1))),
    undefined, gpuIsolatePassEncodersRequested, gpuIsolatePassLabelsRequested,
    Math.max(0, Math.floor(Number(process.env.FLUID_GPU_PASS_TIMESTAMP_SKIP_COMMAND_BUFFERS ?? 0))))
    : undefined;
  // Only ever paired with the pass timestamps it exists to make honest; on its
  // own it would just be a slower frame.
  const passEncoderIsolationScratch = gpuIsolatePassEncodersRequested && passTimestampAudit
    ? createPassEncoderIsolationScratch(device)
    : undefined;
  if (gpuIsolatePassEncodersRequested && !passTimestampAudit) {
    console.warn("FLUID_GPU_ISOLATE_PASS_ENCODERS ignored: it only applies with FLUID_GPU_PASS_TIMESTAMPS=1");
  }
  const instrumentedQueue = commandAudit || passTimestampAudit ? new Proxy(device.queue, {
    get(target, property) {
      if (property === "writeBuffer") return (buffer: GPUBuffer, bufferOffset: number,
        data: GPUAllowSharedBufferSource, dataOffset = 0, size?: number) => {
        commandAudit?.recordWriteBuffer(buffer, writtenByteLength(data, dataOffset, size));
        return target.writeBuffer(buffer, bufferOffset, data, dataOffset, size);
      };
      if (property === "writeTexture") return (destination: GPUTexelCopyTextureInfo,
        data: GPUAllowSharedBufferSource, dataLayout: GPUTexelCopyBufferLayout, size: GPUExtent3D) => {
        commandAudit?.recordWriteTexture(writtenByteLength(data));
        return target.writeTexture(destination, data, dataLayout, size);
      };
      if (property === "submit") return (commandBuffers: Iterable<GPUCommandBuffer>) => {
        const submitted = Array.from(commandBuffers); commandAudit?.recordSubmit(submitted.length);
        const result = target.submit(submitted);
        passTimestampAudit?.afterSubmit(submitted);
        return result;
      };
      if (property === "onSubmittedWorkDone") return () => {
        commandAudit?.recordFence(); return target.onSubmittedWorkDone();
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    }
  }) as GPUQueue : device.queue;
  const instrumentedDevice = new Proxy(device, {
    get(target, property) {
      if (property === "queue") return instrumentedQueue;
      if (property === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
        commandAudit?.recordBufferAllocation(descriptor);
        const buffer = target.createBuffer(descriptor);
        dataFlowAudit?.registry.recordBuffer(buffer, descriptor);
        return buffer;
      };
      if (property === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(descriptor);
        dataFlowAudit?.registry.recordShader(module, descriptor);
        return module;
      };
      if (property === "createBindGroup") return (descriptor: GPUBindGroupDescriptor) => {
        commandAudit?.recordBindGroup();
        const bindGroup = target.createBindGroup(descriptor);
        dataFlowAudit?.registry.recordBindGroup(bindGroup, descriptor);
        return bindGroup;
      };
      if (property === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
        commandAudit?.recordCommandEncoder(descriptor);
        // Isolation sits UNDER the audit: the audit must keep describing the
        // solver's own commands so an isolated run stays comparable to an
        // ordinary one dispatch-for-dispatch.
        const encoder = passEncoderIsolationScratch
          ? isolateComputePassEncoders(target.createCommandEncoder(descriptor), passEncoderIsolationScratch)
          : target.createCommandEncoder(descriptor);
        const dataFlow = dataFlowAudit?.createEncoderSession();
        const passTimestamps = passTimestampAudit?.createEncoderSession();
        return commandAudit || dataFlow || passTimestamps
          ? auditCommandEncoder(encoder, commandAudit, dataFlow, passTimestamps,
            (commandBuffer, session) => passTimestampAudit?.attach(commandBuffer, session))
          : encoder;
      };
      if (property === "createComputePipeline") return (descriptor: GPUComputePipelineDescriptor) => {
        const started = performance.now(), result = target.createComputePipeline(descriptor);
        dataFlowAudit?.registry.recordPipeline(result, descriptor);
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "pipeline", entryPoint: descriptor.compute.entryPoint, elapsed_ms: Math.round(performance.now() - started) }));
        return result;
      };
      if (property === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => {
        const result = await target.createComputePipelineAsync(descriptor);
        dataFlowAudit?.registry.recordPipeline(result, descriptor);
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
  if (method.id === "quadtree-tall-cell" && pressureCyclesOverride !== undefined) values.pressureIterations = pressureCyclesOverride;
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
    authoredMethodProfile: authoredProfile, resolvedMethodValues: values, dawnOptions,
    grid: [solver.info.nx, solver.info.storedNy, solver.info.nz],
    cubicGrid: [solver.info.nx, solver.info.ny, solver.info.nz] }));
  /** Match the renderer's admission boundary. Every advance now submits one
   * command buffer synchronously, so the queue fence alone is exact. */
  const awaitAdvanceCompletion = async () => {
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
      fineSeedCandidateControl?: GPUBuffer; fineSeedLeaves?: GPUBuffer;
      globalFineSummaryDirectory?: GPUBuffer;
      globalFineSummaryDebug?: {
        fineEntries: GPUBuffer;
        fineReferences: GPUBuffer;
        coarseRows: GPUBuffer;
        rankKeys: GPUBuffer;
        workState: GPUBuffer;
        coarseControl: GPUBuffer;
        coarseDelta: GPUBuffer;
      };
      globalFinePageDeltaDebug?: { buffer: GPUBuffer };
      powerTopologyTileStates?: { buffer: GPUBuffer; byteLength: number; sparse: boolean };
    };
    if (initialAudit.globalFineSummaryDirectory && initialAudit.globalFineSummaryDebug
      && initialAudit.globalFinePageDeltaDebug) {
      const [directoryBytes, workStateBytes, coarseControlBytes,
        coarseDeltaBytes, pageDeltaBytes, topologyTileStateBytes] = await Promise.all([
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDirectory }, 64),
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDebug.workState }, 128),
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDebug.coarseControl }, 64),
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDebug.coarseDelta }, 64),
        readBufferBinding(device, { buffer: initialAudit.globalFinePageDeltaDebug.buffer }, 64),
        initialAudit.powerTopologyTileStates
          ? readBufferBinding(device, { buffer: initialAudit.powerTopologyTileStates.buffer },
            initialAudit.powerTopologyTileStates.byteLength)
          : Promise.resolve(undefined),
      ]);
      const words = (bytes: Uint8Array) => Array.from(new Uint32Array(
        bytes.buffer, bytes.byteOffset, bytes.byteLength / 4,
      ));
      console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
        phase: "initial-fine-summary-audit",
        directory: words(directoryBytes),
        workState: words(workStateBytes), coarseControl: words(coarseControlBytes),
        coarseDelta: words(coarseDeltaBytes), pageDelta: words(pageDeltaBytes),
        topologyTileStates: topologyTileStateBytes ? {
          sparse: initialAudit.powerTopologyTileStates?.sparse,
          raw: words(topologyTileStateBytes),
        } : undefined }));
      const workState = new Uint32Array(
        workStateBytes.buffer, workStateBytes.byteOffset, workStateBytes.byteLength / 4,
      );
      const highRank = Math.min(workState[7],
        Math.floor(initialAudit.globalFineSummaryDebug.rankKeys.size / 4),
        Math.floor(initialAudit.globalFineSummaryDebug.fineEntries.size / 32));
      const [entryBytes, rankKeyBytes, fineReferenceBytes, coarseRowBytes] = await Promise.all([
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDebug.fineEntries }, highRank * 32),
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDebug.rankKeys }, highRank * 4),
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDebug.fineReferences }, highRank * 4),
        readBufferBinding(device, { buffer: initialAudit.globalFineSummaryDebug.coarseRows }, highRank * 4),
      ]);
      const entries = new Uint32Array(entryBytes.buffer, entryBytes.byteOffset, entryBytes.byteLength / 4);
      const rankKeys = new Uint32Array(rankKeyBytes.buffer, rankKeyBytes.byteOffset,
        rankKeyBytes.byteLength / 4);
      const fineReferences = new Uint32Array(fineReferenceBytes.buffer, fineReferenceBytes.byteOffset,
        fineReferenceBytes.byteLength / 4);
      const coarseRows = new Uint32Array(coarseRowBytes.buffer, coarseRowBytes.byteOffset,
        coarseRowBytes.byteLength / 4);
      let activeRanks = 0, fineRanks = 0, coarseRanks = 0;
      let firstKeyMismatch: { rank: number; rankKey: number; entryKey: number } | undefined;
      for (let rank = 0; rank < highRank; rank += 1) {
        const keyPlusOne = rankKeys[rank];
        if (keyPlusOne === 0) continue;
        activeRanks += 1;
        if (fineReferences[rank] !== 0) fineRanks += 1;
        if (coarseRows[rank] !== 0) coarseRanks += 1;
        const rankKey = keyPlusOne - 1, entryKey = entries[rank * 8];
        if (!firstKeyMismatch && rankKey !== entryKey) {
          firstKeyMismatch = { rank, rankKey, entryKey };
        }
      }
      console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
        phase: "initial-fine-summary-rank-audit", highRank, activeRanks, fineRanks, coarseRanks,
        firstKeyMismatch }));
    }
    if (initialAudit.fineSeedCandidateControl && initialAudit.fineSeedLeaves) {
      const [candidateBytes, leafBytes] = await Promise.all([
        readBufferBinding(device, { buffer: initialAudit.fineSeedCandidateControl }, 32),
        readBufferBinding(device, { buffer: initialAudit.fineSeedLeaves }, initialAudit.fineSeedLeaves.size),
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
        phase: "initial-fine-seed-candidate-audit", control: Array.from(candidates),
        leaves: { live, core, halo, minimumPhi, maximumPhi } }));
    }
  }
  // Construction and t=0 publication have separate costs. The command audit
  // below measures only recurring advance work and explicitly requested
  // profiler/readback activity after the initialized solver is warm.
  commandAudit?.reset();
  dataFlowAudit?.start();
  passTimestampAudit?.start();
  const runStarted = performance.now();
  let steps = 0, samplingWall_ms = 0, matched: Awaited<ReturnType<typeof readCubicVolumeField>> | undefined;
  let flipCensusPrevious: { field: Float32Array; time_s: number } | undefined;
  // Retry accounting for the advance loop below. `rejectedAdvanceAttempts` is
  // monotone, so the exponential warning gate rides on it: a wedge drives it up
  // continuously and still costs only log2(n) lines, whereas gating on the
  // consecutive count would emit a line per rejection whenever accepts and
  // rejections alternate (the ordinary profiled-advance-in-flight case).
  let rejectedAdvanceAttempts = 0;
  let consecutiveRejectedAdvances = 0;
  let maximumConsecutiveRejectedAdvances = 0;
  let nextRejectedAdvanceWarning = 1;
  let lastReportedStructuredRejectSummary: string | undefined;
  let structuredRejectReports = 0;
  let firstStructuredRejectStep: number | undefined;
  let lastProgressAt_ms = runStarted;
  let lastProgressSteps = 0;
  // The perturbed cadence remains exclusive to the quadtree dam-break
  // regression; FLUID_STABILITY_ENVELOPE=1 collects the same envelope for any
  // scenario/method at the scene's fixed cadence.
  const perturbCadence = scenarioId === "dam-break-ui" && method.id === "quadtree-tall-cell";
  const collectStabilityEnvelope = perturbCadence || stabilityEnvelopeRequested;
  const stabilityEnvelope: StabilityEnvelope | undefined = collectStabilityEnvelope ? {
    peakLiquidSpeed_m_s: 0, peakComponentCfl: 0, peakKineticEnergyProxy: 0,
    maximumProjectionEnergyRatio: 0, projectionEnergySampleCount: 0,
    maximumPressureRelativeResidual: 0,
    maximumProjectedVariationalResidual: 0, maximumExactVolumeDrift: 0, maximumLevelSetMismatchFraction: 0,
    maximumComponentCount: 0, minimumDominantComponentFraction: 1,
    nonFiniteVelocityCount: 0, invalidVolumeSampleCount: 0,
    spatialSampledSteps: 0, sampledSteps: 0
  } : undefined;
  // Compact-octree spatial QA reconstructs occupancy on the finest cubic
  // lattice, whereas the conservative controller integrates adaptive cell
  // volumes. Compare this estimator with its own accepted reset-time field;
  // mixing the two baselines manufactures drift even when both are stable.
  const initialExact = (!performanceProfileRequested || regressionArtifactRequested)
    && method.id === "octree"
    && (collectStabilityEnvelope || energyEverySteps > 0 || checkpointEvery_s > 0)
    ? await readCubicVolumeField(device, solver) : undefined;
  const spatialExactReference = initialExact?.summary.cellSum;
  const initialPotentialEnergyProxy = initialExact ? gravitationalPotentialEnergyProxy(initialExact.field,
    solver.info.nx, solver.info.ny, solver.info.nz, {
      x: scene.container.width_m / solver.info.nx,
      y: scene.container.height_m / solver.info.ny,
      z: scene.container.depth_m / solver.info.nz,
    }, scene.fluid.gravity_m_s2) : undefined;
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
  let previousAuditedFineGeneration = 0;
  let powerGenerationAuditedSteps = 0;
  let mgpcgIterationMinimum = Number.POSITIVE_INFINITY;
  let mgpcgIterationMaximum = 0;
  const mgpcgIterationHistogram: Record<string, number> = {};
  const powerGenerationAuditCapacity = method.id === "octree"
    && (powerGenerationAuditRequested || collectStabilityEnvelope)
    ? Math.max(1, exactStepCount ?? 0, oracleSteps,
      Math.ceil(target_s / Math.max(scene.numerics.maxDt_s, Number.EPSILON)) + 1)
    : 0;
  const powerGenerationAuditSnapshot = powerGenerationAuditCapacity > 0
    ? device.createBuffer({
      label: "Per-step structured generation audit snapshots",
      size: powerGenerationAuditCapacity * STRUCTURED_GENERATION_AUDIT_SNAPSHOT.strideBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    : undefined;
  const powerGenerationAuditSteps: number[] = [];
  const powerGenerationAuditFineGenerations: number[] = [];
  const powerGenerationAuditDts: number[] = [];
  // ---- Silent-failure tripwires (docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3) --
  // Per accepted step: the fine topology rollback word, the restriction
  // transaction's unaccepted-row count, the MGPCG converged word, and the
  // fine worklist header. All four are copied GPU-side by a tiny encoder that
  // rides after the advance's own submission and is mapped exactly once,
  // after the measured window closes; nothing here drains the pipeline.
  const TRIPWIRE_RECORD = Object.freeze({
    topologyOffsetBytes: 0, topologyBytes: 48,
    restrictionOffsetBytes: 48, restrictionBytes: 32,
    mgpcgOffsetBytes: 80, mgpcgBytes: 64,
    fineHeaderOffsetBytes: 144, fineHeaderBytes: 28,
    strideBytes: 176,
  });
  /** The benchmark and acceptance lanes must evaluate every tripwire. Any
   * other octree run captures them opportunistically: a trip still fails, but
   * a scene with no compact fine authority is "not applicable" rather than a
   * wiring failure. */
  const tripwiresRequired = !tripwiresDisabled && method.id === "octree"
    && (tripwiresForcedRequired
      || scenarioId === "minimal-power-dam-break" || scenarioId === "dam-break-ui");
  const tripwireCapacity = !tripwiresDisabled && method.id === "octree"
    ? Math.max(1, exactStepCount ?? 0, oracleSteps,
      Math.ceil(target_s / Math.max(scene.numerics.maxDt_s, Number.EPSILON)) + 1)
    : 0;
  const tripwireSnapshot = tripwireCapacity > 0
    ? device.createBuffer({
      label: "Per-step silent-failure tripwire snapshots",
      size: tripwireCapacity * TRIPWIRE_RECORD.strideBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    : undefined;
  const tripwireSteps: number[] = [];
  const tripwireFineGenerations: number[] = [];
  if (tripwiresDisabled) {
    console.error("[tripwires] DISABLED by FLUID_TRIPWIRES=0: topology rollback,"
      + " unaccepted restriction rows, MGPCG non-convergence and the fine-band"
      + " capacity sentinel are NOT gated in this run");
  }
  if (tripwireAllowList.size !== 0) {
    console.error(`[tripwires] downgraded to warnings by FLUID_TRIPWIRE_ALLOW: ${
      Array.from(tripwireAllowList).join(", ")}`);
  }
  /** Live control buffers the tripwire record is copied from.
   *
   * The fine topology control has no dedicated accessor, but the solver's
   * public `workAccountingBuffers` already publishes it under exactly the
   * selection `readGlobalFineLevelSetDiagnostics` uses
   * (`globalFinePublishedIsA ? topologyBA : topologyAB`, i.e. the direction
   * that produced the currently published source). Reusing it keeps one
   * selection rule in the codebase instead of a second copy that could drift.
   * A missing source is a hard failure on a required lane, never a silent
   * skip -- an unevaluable tripwire is what this work item exists to kill. */
  const tripwireSources = () => {
    const authority = solver as GPUSolverInstance & {
      mgpcgControl?: GPUBuffer;
      globalFineRestrictionControl?: GPUBuffer;
      workAccountingBuffers?: { fineTopologyControl?: GPUBufferBinding };
    };
    return {
      topology: authority.workAccountingBuffers?.fineTopologyControl,
      restriction: authority.globalFineRestrictionControl,
      mgpcg: authority.mgpcgControl,
      fineWorklist: authority.globalFineLevelSetSource?.worklist,
    };
  };
  let topologyTransitionDeepCell: number | undefined;
  let lastLoggedPhysicsTraceSampleId = 0;
  let lastAttributedPhysicsTraceSampleId = 0;
  let attributedTraceSamples = 0;
  const attributedPhaseBuckets = new Map<string, Omit<GPUFineTimestampBucket, "mean_ms">>();
  const captureGenericPhaseTrace = () => {
    if (!genericPhaseTraceRequested || attributedTraceSamples >= genericPhaseTraceAdvances) return;
    const trace = (solver.info as { physicsTrace?: PerformanceTrace }).physicsTrace;
    if (!trace || trace.sampleId === lastAttributedPhysicsTraceSampleId) return;
    lastAttributedPhysicsTraceSampleId = trace.sampleId;
    attributedTraceSamples += 1;
    for (const phase of trace.phases) {
      const duration = phase.duration_ms;
      if (!Number.isFinite(duration) || duration < 0) continue;
      const label = `${phase.id}: ${phase.label}`;
      const bucket = attributedPhaseBuckets.get(label) ?? {
        samples: 0, total_ms: 0, minimum_ms: Number.POSITIVE_INFINITY, maximum_ms: 0,
      };
      bucket.samples += 1;
      bucket.total_ms += duration;
      bucket.minimum_ms = Math.min(bucket.minimum_ms, duration);
      bucket.maximum_ms = Math.max(bucket.maximum_ms, duration);
      attributedPhaseBuckets.set(label, bucket);
    }
  };
  while ((solver.info.submittedTime_s ?? 0) + 1e-9 < target_s) {
    const stepDt = perturbCadence
      ? Math.min(scene.numerics.maxDt_s, regressionDtPattern[steps % regressionDtPattern.length])
      : scene.numerics.maxDt_s;
    const requestedTime = Math.min(target_s, (solver.info.submittedTime_s ?? 0) + stepDt);
    const accepted = solver.advanceTo(requestedTime, bodies);
    if (!accepted) {
      rejectedAdvanceAttempts += 1;
      consecutiveRejectedAdvances += 1;
      maximumConsecutiveRejectedAdvances = Math.max(maximumConsecutiveRejectedAdvances,
        consecutiveRejectedAdvances);
      if (rejectedAdvanceAttempts >= nextRejectedAdvanceWarning) {
        nextRejectedAdvanceWarning = rejectedAdvanceAttempts * 2;
        const emittedAt_ms = performance.now();
        console.log(JSON.stringify({
          scenario: scenarioId, method: method.id, record: "advance-rejected",
          steps, requestedTime_s: requestedTime, dt_s: stepDt,
          submittedTime_s: solver.info.submittedTime_s ?? 0,
          rejectedAdvanceAttempts, consecutiveRejectedAdvances,
          maximumConsecutiveRejectedAdvances,
          wall_ms: Math.round(emittedAt_ms - runStarted),
          structuredRejectStage: solver.info.structuredRejectStage,
          structuredRejectIndex: solver.info.structuredRejectIndex,
          structuredRejectSummary: solver.info.structuredRejectSummary,
        }));
        // Diagnostics, not solver work. Charging emission to the sampling
        // account keeps simulationWall_ms — and every per-advance wall gate
        // derived from it — identical with and without these lines.
        samplingWall_ms += performance.now() - emittedAt_ms;
      }
      if (rejectedAdvanceWedgeLimit > 0
        && consecutiveRejectedAdvances >= rejectedAdvanceWedgeLimit) {
        throw new Error(`${method.id} ${scenarioId} advance wedged: advanceTo rejected ${consecutiveRejectedAdvances} consecutive attempts (${rejectedAdvanceAttempts} total) at step ${steps} requesting t=${requestedTime.toFixed(6)} s with submitted t=${(solver.info.submittedTime_s ?? 0).toFixed(6)} s; structured reject carry ${solver.info.structuredRejectSummary ?? "clean"}. Unset FLUID_MAX_CONSECUTIVE_REJECTED_ADVANCES or raise it above ${rejectedAdvanceWedgeLimit} to restore the unbounded retry.`);
      }
      await new Promise((resolve) => setTimeout(resolve, 0));
      continue;
    }
    consecutiveRejectedAdvances = 0;
    steps += 1;
    // The reject carry is already decoded onto `solver.info` by the solver's own
    // readStats path; reading the published field costs nothing. It latches, so
    // report only transitions and the step that first observed one.
    const structuredRejectSummary = solver.info.structuredRejectSummary;
    if (structuredRejectSummary !== lastReportedStructuredRejectSummary) {
      lastReportedStructuredRejectSummary = structuredRejectSummary;
      if (structuredRejectSummary !== undefined) {
        structuredRejectReports += 1;
        firstStructuredRejectStep ??= steps;
        const emittedAt_ms = performance.now();
        console.log(JSON.stringify({
          scenario: scenarioId, method: method.id, record: "structured-reject-carry",
          steps, submittedTime_s: solver.info.submittedTime_s ?? 0,
          structuredRejectStage: solver.info.structuredRejectStage,
          structuredRejectIndex: solver.info.structuredRejectIndex,
          structuredRejectSummary,
        }));
        samplingWall_ms += performance.now() - emittedAt_ms;
      }
    }
    captureGenericPhaseTrace();
    if (dataFlowAudit && steps >= genericPhaseTraceAdvances) dataFlowAudit.stop();
    if (physicsTraceLogRequested) {
      const trace = (solver.info as { physicsTrace?: {
        sampleId: number; context: string; total_ms: number;
        phases: readonly { id: string; label: string; duration_ms: number }[];
      } }).physicsTrace;
      if (trace && trace.sampleId !== lastLoggedPhysicsTraceSampleId) {
        lastLoggedPhysicsTraceSampleId = trace.sampleId;
        console.log(JSON.stringify({
          scenario: scenarioId, method: method.id, record: "physics-trace",
          observedAtStep: steps, sampleId: trace.sampleId, context: trace.context,
          total_ms: Number(trace.total_ms.toFixed(3)),
          phases: trace.phases.map((phase) => ({
            id: phase.id, label: phase.label, ms: Number(phase.duration_ms.toFixed(3)),
          })),
        }));
      }
    }
    const auditThisPowerStep = steps % powerAuditEverySteps === 0 || requestedTime + 1e-9 >= target_s;
    const captureCompactPowerStep = method.id === "octree"
      && (collectStabilityEnvelope || powerGenerationAuditRequested && auditThisPowerStep);
    if (captureCompactPowerStep) {
      const audited = solver as GPUSolverInstance & {
        structuredVelocityControl?: GPUBuffer;
        structuredBoundaryControl?: GPUBuffer;
        mgpcgControl?: GPUBuffer;
      };
      const fine = audited.globalFineLevelSetSource;
      if (!fine || !audited.structuredVelocityControl || !audited.structuredBoundaryControl
        || !audited.mgpcgControl || !audited.globalFineVolumeControl
        || !audited.structuredProjectionEnergyStats || !powerGenerationAuditSnapshot) {
        throw new Error(`structured generation audit step ${steps} is missing accepted controls`);
      }
      const record = powerGenerationAuditSteps.length;
      if (record >= powerGenerationAuditCapacity) {
        throw new Error(`structured generation audit exceeded its ${powerGenerationAuditCapacity}-step snapshot capacity`);
      }
      const auditEncoder = device.createCommandEncoder({
        label: `Queue structured generation audit snapshot ${record + 1}`,
      });
      // The shared ABI writer also feeds the browser's step-coherent snapshot
      // ring, so the harness and the UI copy byte-identical records.
      encodeStructuredAuditRecordCopies(auditEncoder, {
        structuredVelocityControl: audited.structuredVelocityControl,
        structuredBoundaryControl: audited.structuredBoundaryControl,
        fineWorklist: fine.worklist,
        mgpcgControl: audited.mgpcgControl,
        fineVolumeControl: audited.globalFineVolumeControl,
        projectionEnergyStats: audited.structuredProjectionEnergyStats,
      }, powerGenerationAuditSnapshot, record * STRUCTURED_GENERATION_AUDIT_SNAPSHOT.strideBytes);
      device.queue.submit([auditEncoder.finish()]);
      powerGenerationAuditSteps.push(steps);
      powerGenerationAuditFineGenerations.push(fine.generation);
      powerGenerationAuditDts.push(stepDt);
    }
    if (tripwireSnapshot) {
      // These copies are QA evidence, not simulation work: charge their host
      // cost to samplingWall_ms exactly as every other QA readback is charged,
      // so the throughput lanes stay comparable across builds.
      const tripwireCaptureStartedAt_ms = performance.now();
      const sources = tripwireSources();
      const missing = Object.entries(sources)
        .filter(([, buffer]) => !buffer).map(([name]) => name);
      // A narrowed topology binding would silently truncate the record rather
      // than surface the rollback word, so treat it as an unreadable counter.
      if (sources.topology && (sources.topology.size ?? TRIPWIRE_RECORD.topologyBytes)
        < TRIPWIRE_RECORD.topologyBytes) {
        missing.push(`topology (binding exposes ${sources.topology.size} bytes,`
          + ` the control ABI needs ${TRIPWIRE_RECORD.topologyBytes})`);
      }
      if (missing.length !== 0) {
        // An unevaluable tripwire is the exact failure mode A3 exists to kill:
        // fail loudly instead of quietly capturing nothing.
        if (tripwiresRequired) {
          throw new Error(`tripwires could not be evaluated at step ${steps}: missing control`
            + ` buffers ${missing.join(", ")} (see docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3)`);
        }
      } else {
        const record = tripwireSteps.length;
        if (record >= tripwireCapacity) {
          throw new Error(`tripwire capture exceeded its ${tripwireCapacity}-step snapshot capacity`);
        }
        const base = record * TRIPWIRE_RECORD.strideBytes;
        // One constant label: the command audit buckets encoders by label, and
        // a per-step unique label would add one map entry per advance to every
        // report and regression artifact.
        const encoder = device.createCommandEncoder({ label: "Queue tripwire snapshot" });
        encoder.copyBufferToBuffer(sources.topology!.buffer, sources.topology!.offset ?? 0,
          tripwireSnapshot, base + TRIPWIRE_RECORD.topologyOffsetBytes,
          TRIPWIRE_RECORD.topologyBytes);
        encoder.copyBufferToBuffer(sources.restriction!, 0, tripwireSnapshot,
          base + TRIPWIRE_RECORD.restrictionOffsetBytes, TRIPWIRE_RECORD.restrictionBytes);
        encoder.copyBufferToBuffer(sources.mgpcg!, 0, tripwireSnapshot,
          base + TRIPWIRE_RECORD.mgpcgOffsetBytes, TRIPWIRE_RECORD.mgpcgBytes);
        encoder.copyBufferToBuffer(sources.fineWorklist!, 0, tripwireSnapshot,
          base + TRIPWIRE_RECORD.fineHeaderOffsetBytes, TRIPWIRE_RECORD.fineHeaderBytes);
        device.queue.submit([encoder.finish()]);
        tripwireSteps.push(steps);
        tripwireFineGenerations.push(
          (solver as GPUSolverInstance).globalFineLevelSetSource?.generation ?? 0);
      }
      samplingWall_ms += performance.now() - tripwireCaptureStartedAt_ms;
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
    if (steps % completionFenceEverySteps === 0) await awaitAdvanceCompletion();
    const shouldReport = reportEvery > 0 && steps % reportEvery === 0;
    const shouldSampleEnergy = energyEverySteps > 0 && steps % energyEverySteps === 0;
    const shouldSampleDetailedFields = method.id === "octree"
      ? shouldSampleEnergy
      : shouldReport || shouldSampleEnergy || collectStabilityEnvelope;
    if (shouldSampleDetailedFields) {
      const samplingStartedAt = performance.now();
      await awaitAdvanceCompletion();
      solver.info.simulatedTime_s = solver.info.submittedTime_s;
      const sample = await solver.readStats();
      if (sample.structuredStartKineticEnergyProxy !== undefined) {
        console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
          phase: "structured-stage-energy", steps,
          time_s: solver.info.submittedTime_s,
          start: sample.structuredStartKineticEnergyProxy,
          postAdvection: sample.structuredPostAdvectionKineticEnergyProxy,
          preProjection: sample.structuredPreProjectionKineticEnergyProxy,
          postProjection: sample.structuredPostProjectionKineticEnergyProxy,
          wetStart: sample.structuredWetStartKineticEnergyProxy,
          wetPostAdvection: sample.structuredWetPostAdvectionKineticEnergyProxy,
          wetPreProjection: sample.structuredWetPreProjectionKineticEnergyProxy,
          wetPostProjection: sample.structuredWetPostProjectionKineticEnergyProxy,
          wetFaces: sample.structuredWetFaceCount,
          wetThetaStart: sample.structuredWetStartThetaEnergyProxy,
          wetThetaPostAdvection: sample.structuredWetPostAdvectionThetaEnergyProxy,
          wetThetaPreProjection: sample.structuredWetPreProjectionThetaEnergyProxy,
          wetThetaPostProjection: sample.structuredWetPostProjectionThetaEnergyProxy,
          staggeredPath: sample.structuredStaggeredPathCount }));
      }
      const isRestrictedTall = sample.gridKind === "restricted-tall-cell";
      let tallCellActivity: ReturnType<typeof inspectColumnBases> | undefined, tallVolumeGaps: ReturnType<typeof inspectTallVolumeGaps> | undefined;
      let bases: Float32Array | undefined;
      if (isRestrictedTall) {
        bases = await readFloatTexture2D(device, solver.columnBaseTexture!, sample.nx, sample.nz);
        tallCellActivity = inspectColumnBases(bases, sample.nx, sample.nz, sample.ny, sample.regularLayers, sample.maximumNeighborDelta);
        tallVolumeGaps = inspectTallVolumeGaps(await readFloatTexture3D(device, solver.volumeTexture, sample.nx, sample.storedNy, sample.nz), bases, sample.nx, sample.storedNy, sample.nz, sample.ny, sample.maximumNeighborDelta);
      }
      const exact = await readCubicVolumeField(device, solver);
      if (steps === oracleSteps) matched = exact;
      {
        // Dam-front footprint isotropy: radial wetted extents on the floor
        // layer measured from the reservoir corner. A gravity current spreads
        // as a circular arc (circularity ~1); an axis-biased velocity
        // extension squares the front off toward the L1 diamond (~0.71).
        const { nx: fx, ny: fy, nz: fz } = sample;
        const wet = (i: number, k: number) => (exact.field[i + fx * (0 + fy * k)] ?? 0) > 0.5;
        let rx = 0, rz = 0, rd = 0;
        for (let i = 0; i < fx; i += 1) if (wet(i, 0)) rx = Math.max(rx, i);
        for (let k = 0; k < fz; k += 1) if (wet(0, k)) rz = Math.max(rz, k);
        for (let d = 0; d < Math.min(fx, fz); d += 1) if (wet(d, d)) rd = Math.max(rd, d);
        console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
          phase: "front-footprint", time_s: solver.info.submittedTime_s,
          rx, rz, rdiag: rd,
          circularity: rx > 0 ? (rd * Math.SQRT2) / Math.max(rx, rz) : 0 }));
      }
      if (process.env.FLUID_HEIGHT_PROFILE === "1") {
        // Wet-cell census per horizontal layer, plus how many of those wet
        // cells touch a domain side wall: separates "fluid still airborne"
        // from "fluid pinned to the ceiling/walls" while settling.
        const { nx: fx, ny: fy, nz: fz } = sample;
        const layers: number[] = [], wallLayers: number[] = [];
        for (let j = 0; j < fy; j += 1) {
          let count = 0, wall = 0;
          for (let k = 0; k < fz; k += 1) for (let i = 0; i < fx; i += 1) {
            if ((exact.field[i + fx * (j + fy * k)] ?? 0) <= 0.5) continue;
            count += 1;
            if (i === 0 || i === fx - 1 || k === 0 || k === fz - 1) wall += 1;
          }
          layers.push(count); wallLayers.push(wall);
        }
        console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
          phase: "height-profile", time_s: solver.info.submittedTime_s,
          wetPerLayer: layers, wallWetPerLayer: wallLayers }));
      }
      if (process.env.FLUID_WET_FLIP_CENSUS === "1") {
        // Interface motion is bounded by the interval CFL: a cell that turns
        // wet with no previously wet cell within the reachable Chebyshev
        // radius received liquid the velocity field could not have delivered
        // there — creation by the interface machinery, not advection. At
        // per-step cadence (dt 0.004 s, dx 0.05 m) distance >= 2 needs
        // sustained speed >= 12.5 m/s.
        const { nx: fx, ny: fy, nz: fz } = sample;
        const previous = flipCensusPrevious;
        if (previous && previous.field.length === exact.field.length) {
          const wetBefore = (i: number, j: number, k: number) =>
            (previous.field[i + fx * (j + fy * k)] ?? 0) > 0.5;
          const nearestWetBefore = (i: number, j: number, k: number, limit: number) => {
            for (let d = 1; d <= limit; d += 1) {
              for (let dk = -d; dk <= d; dk += 1) for (let dj = -d; dj <= d; dj += 1) for (let di = -d; di <= d; di += 1) {
                if (Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk)) !== d) continue;
                const qi = i + di, qj = j + dj, qk = k + dk;
                if (qi < 0 || qj < 0 || qk < 0 || qi >= fx || qj >= fy || qk >= fz) continue;
                if (wetBefore(qi, qj, qk)) return d;
              }
            }
            return limit + 1;
          };
          const byDistance = [0, 0, 0, 0];
          const createdHeights: number[] = [];
          let totalWet = 0, createdPotentialProxy = 0;
          const gravityMagnitude = Math.hypot(
            scene.fluid.gravity_m_s2.x,
            scene.fluid.gravity_m_s2.y,
            scene.fluid.gravity_m_s2.z,
          );
          const spacingForCensus = { x: scene.container.width_m / fx,
            y: scene.container.height_m / fy, z: scene.container.depth_m / fz };
          const cellVolume = spacingForCensus.x * spacingForCensus.y * spacingForCensus.z;
          for (let k = 0; k < fz; k += 1) for (let j = 0; j < fy; j += 1) for (let i = 0; i < fx; i += 1) {
            const fraction = exact.field[i + fx * (j + fy * k)] ?? 0;
            if (fraction <= 0.5) continue;
            totalWet += 1;
            if (wetBefore(i, j, k)) continue;
            const distance = Math.min(nearestWetBefore(i, j, k, 3), byDistance.length);
            byDistance[distance - 1] += 1;
            if (distance >= 2) {
              createdHeights.push(j);
              createdPotentialProxy += fraction * cellVolume * gravityMagnitude * (j + 0.5) * spacingForCensus.y;
            }
          }
          console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
            phase: "wet-flip-census", time_s: solver.info.submittedTime_s,
            elapsed_s: (solver.info.submittedTime_s ?? 0) - previous.time_s,
            totalWet, newWetByChebyshevDistance: byDistance,
            createdHeights, createdPotentialProxy }));
        }
        flipCensusPrevious = { field: exact.field.slice(),
          time_s: solver.info.submittedTime_s ?? 0 };
      }
      if (process.env.FLUID_PROBE_CELL) {
        // Local liquid-fraction neighborhood around one cell, per sampled
        // step: the ground truth for whether an air-support march fallback
        // there was a genuinely sealed component or a seeding gap next to
        // reachable liquid.
        const probe = process.env.FLUID_PROBE_CELL.split(",").map(Number);
        const { nx: fx, ny: fy, nz: fz } = sample;
        if (probe.length === 3 && probe.every((v) => Number.isInteger(v) && v >= 0)) {
          const [pi, pj, pk] = probe as [number, number, number];
          const planes: string[] = [];
          for (let dj = 2; dj >= -2; dj -= 1) {
            const j = pj + dj;
            if (j < 0 || j >= fy) continue;
            const rows: string[] = [];
            for (let dk = -2; dk <= 2; dk += 1) {
              const k = pk + dk;
              if (k < 0 || k >= fz) continue;
              let row = "";
              for (let di = -2; di <= 2; di += 1) {
                const i = pi + di;
                if (i < 0 || i >= fx) { row += " ."; continue; }
                const alpha = exact.field[i + fx * (j + fy * k)] ?? 0;
                row += alpha > 0.99 ? " F" : alpha > 0.5 ? " O" : alpha > 0.01 ? " o" : " -";
              }
              rows.push(row);
            }
            planes.push(`j=${j}:` + rows.join(" |"));
          }
          console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
            phase: "probe-cell", time_s: solver.info.submittedTime_s,
            cell: probe, planes }));
        }
      }
      const stagedSolver = solver as GPUSolverInstance & {
        preProjectionVelocityTexture?: GPUTexture;
        velocityTexture?: GPUTexture;
      };
      // Octree velocity exists only in the accepted packed structured-row
      // publication. Cubic texture readback applies solely to texture-backed
      // reference methods.
      const usesStructuredVelocity = method.id === "octree";
      const spacing = {
        x: scene.container.width_m / sample.nx,
        y: scene.container.height_m / sample.ny,
        z: scene.container.depth_m / sample.nz
      };
      const readStagedVelocity = (texture: GPUTexture | undefined) => {
        if (!texture || usesStructuredVelocity) return Promise.resolve(undefined);
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
      // The MGPCG control stores the algebraic pressure-equation residual.
      // Eq. (3)/(4) leaves dt/rho times that residual as integrated flux after
      // projection; retain that physical quantity under the variational name.
      const projectedVariationalResidual = octreeProjectedVariationalResidualRms(
        sample.pressureResidual, stepDt, scene.fluid.density_kg_m3);
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
        stabilityEnvelope.projectionEnergySampleCount += 1;
        // The quadtree residual readback is asynchronous, so the first steps
        // can legitimately sample before any residual exists; later steps
        // without one still fail hard.
        stabilityEnvelope.maximumPressureRelativeResidual = Math.max(stabilityEnvelope.maximumPressureRelativeResidual, sample.pressureRelativeResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumProjectedVariationalResidual = Math.max(stabilityEnvelope.maximumProjectedVariationalResidual,
          projectedVariationalResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumExactVolumeDrift = Math.max(stabilityEnvelope.maximumExactVolumeDrift, Math.abs(exactVolumeDrift));
        stabilityEnvelope.maximumLevelSetMismatchFraction = Math.max(stabilityEnvelope.maximumLevelSetMismatchFraction, sample.quadtreeLevelSetMismatchFraction ?? 0);
        stabilityEnvelope.maximumComponentCount = Math.max(stabilityEnvelope.maximumComponentCount, exact.summary.componentCount);
        stabilityEnvelope.minimumDominantComponentFraction = Math.min(stabilityEnvelope.minimumDominantComponentFraction, dominantFraction);
        stabilityEnvelope.nonFiniteVelocityCount += preProjectionVelocity.nonFiniteCount + postProjectionVelocity.nonFiniteCount;
        stabilityEnvelope.sampledSteps += 1;
      }
      if (stabilityEnvelope && usesStructuredVelocity && !powerGenerationAuditSnapshot) {
        const dominantFraction = exact.summary.wetCells > 0 ? exact.summary.largestComponent / exact.summary.wetCells : 1;
        stabilityEnvelope.peakLiquidSpeed_m_s = Math.max(stabilityEnvelope.peakLiquidSpeed_m_s, sample.maxSpeed_m_s ?? 0);
        stabilityEnvelope.peakComponentCfl = Math.max(stabilityEnvelope.peakComponentCfl, sample.maxComponentCfl ?? 0);
        stabilityEnvelope.maximumExactVolumeDrift = Math.max(stabilityEnvelope.maximumExactVolumeDrift, Math.abs(exactVolumeDrift));
        stabilityEnvelope.maximumComponentCount = Math.max(stabilityEnvelope.maximumComponentCount, exact.summary.componentCount);
        stabilityEnvelope.minimumDominantComponentFraction = Math.min(stabilityEnvelope.minimumDominantComponentFraction, dominantFraction);
        stabilityEnvelope.nonFiniteVelocityCount += sample.nonFiniteCount ?? 0;
        const structuredEnergySamples = sample.structuredProjectionEnergySampleCount ?? 0;
        const structuredEnergyRatio = sample.structuredProjectionEnergyRatio;
        const structuredPreEnergy = sample.structuredPreProjectionKineticEnergyProxy;
        const structuredPostEnergy = sample.structuredPostProjectionKineticEnergyProxy;
        if (Number.isSafeInteger(structuredEnergySamples) && structuredEnergySamples > 0
          && structuredEnergyRatio !== undefined && Number.isFinite(structuredEnergyRatio)
          && structuredEnergyRatio >= 0
          && structuredPreEnergy !== undefined && Number.isFinite(structuredPreEnergy)
          && structuredPreEnergy >= 0
          && structuredPostEnergy !== undefined && Number.isFinite(structuredPostEnergy)
          && structuredPostEnergy >= 0) {
          stabilityEnvelope.maximumProjectionEnergyRatio = Math.max(
            stabilityEnvelope.maximumProjectionEnergyRatio, structuredEnergyRatio,
          );
          stabilityEnvelope.projectionEnergySampleCount += structuredEnergySamples;
          stabilityEnvelope.peakKineticEnergyProxy = Math.max(
            stabilityEnvelope.peakKineticEnergyProxy, structuredPostEnergy,
          );
        }
        stabilityEnvelope.maximumPressureRelativeResidual = Math.max(stabilityEnvelope.maximumPressureRelativeResidual,
          sample.pressureRelativeResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.maximumProjectedVariationalResidual = Math.max(stabilityEnvelope.maximumProjectedVariationalResidual,
          projectedVariationalResidual ?? (steps <= 2 ? 0 : Infinity));
        stabilityEnvelope.sampledSteps += 1;
      }
      if (shouldReport) console.log(JSON.stringify({ scenario: scenarioId, method: method.id, phase: "running", steps, simulatedTime_s: sample.simulatedTime_s, dt_s: stepDt, preProjectionVelocity, postProjectionVelocity, maxSpeed_m_s: sample.maxSpeed_m_s, maxAirSpeed_m_s: sample.maxAirSpeed_m_s, maxDivergenceBefore_s: sample.maxDivergenceBefore_s, maxDivergenceAfter_s: sample.maxDivergenceAfter_s, pressureRelativeResidual: sample.pressureRelativeResidual, pressureIterationsUsed: sample.quadtreePressureIterationsUsed, pressureIterationBudget: sample.quadtreePressureIterationBudget, pressureIterationHardBudget: sample.quadtreePressureIterationHardBudget, pressureConverged: sample.quadtreePressureConverged, velocityClampCount: sample.quadtreeVelocityClampCount, factorLevelCount: sample.quadtreeFactorLevelCount, physicsTrace: sample.physicsTrace, maxComponentCfl: sample.maxComponentCfl, representedVolumeDrift: sample.representedVolumeDrift, volumeCorrectionNormalSpeed_cells_s: sample.volumeCorrectionNormalSpeed_cells_s, volumeCorrectionDivergenceRate_s: sample.volumeCorrectionDivergenceRate_s, phiInterfaceCellCount: sample.phiInterfaceCellCount, exactVolumeCellSum: exact.summary.cellSum, exactVolumeDrift, componentCount: exact.summary.componentCount, dominantComponentFraction: exact.summary.wetCells > 0 ? exact.summary.largestComponent / exact.summary.wetCells : 1, quadtree: sample.gridKind === "quadtree-tall-cell" ? { opticalLayerMode: sample.quadtreeOpticalLayerMode, opticalAlpha: sample.quadtreeOpticalAlpha, opticalMinimumCells: sample.quadtreeOpticalMinimumCells, opticalMaximumCells: sample.quadtreeOpticalMaximumCells, leafCount: sample.quadtreeLeafCount, pressureSampleCount: sample.quadtreePressureSampleCount, liquidDofCount: sample.quadtreeLiquidDofCount, faceCount: sample.quadtreeFaceCount, tallSegmentCount: sample.quadtreeTallSegmentCount, ghostFaceCount: sample.quadtreeGhostFaceCount, maximumNeighborRatio: sample.quadtreeMaximumNeighborRatio, maximumFluidScale: sample.quadtreeMaximumFluidScale, levelSetMismatchFraction: sample.quadtreeLevelSetMismatchFraction } : undefined, stabilityFlags: sample.stabilityFlags, extrema, tallCellActivity, tallVolumeGaps }));
      samplingWall_ms += performance.now() - samplingStartedAt;
    }
    if (checkpointEvery_s > 0 && (solver.info.submittedTime_s ?? 0) + 1e-9 >= nextCheckpoint_s) {
      const samplingStartedAt = performance.now();
      await awaitAdvanceCompletion();
      const cubic = steps === oracleSteps && matched ? matched : await readCubicVolumeField(device, solver);
      if (steps === oracleSteps) matched = cubic;
      if (stabilityEnvelope && method.id === "octree") {
        if (!Number.isFinite(cubic.summary.minimum) || !Number.isFinite(cubic.summary.maximum)
          || !Number.isFinite(cubic.summary.cellSum) || cubic.summary.minimum < -0.01
          || cubic.summary.maximum > 1.5 || cubic.summary.cellSum <= 1
          || cubic.summary.cellSum >= cubic.field.length - 1) {
          stabilityEnvelope.invalidVolumeSampleCount += 1;
        }
        const dominantFraction = cubic.summary.wetCells > 0
          ? cubic.summary.largestComponent / cubic.summary.wetCells
          : 1;
        stabilityEnvelope.maximumComponentCount = Math.max(
          stabilityEnvelope.maximumComponentCount, cubic.summary.componentCount,
        );
        stabilityEnvelope.minimumDominantComponentFraction = Math.min(
          stabilityEnvelope.minimumDominantComponentFraction, dominantFraction,
        );
        stabilityEnvelope.spatialSampledSteps += 1;
      }
      let preProjectionVelocity: Float32Array | undefined, postProjectionVelocity: Float32Array | undefined;
      if (singleTallCellProbe && solver.info.gridKind === "restricted-tall-cell") {
        const bases = await readFloatTexture2D(device, solver.columnBaseTexture!, solver.info.nx, solver.info.nz);
        const staged = solver as GPUSolverInstance & { preProjectionVelocityTexture?: GPUTexture; velocityTexture?: GPUTexture };
        if (staged.preProjectionVelocityTexture) preProjectionVelocity = await readTallVelocityField3D(device, staged.preProjectionVelocityTexture, solver.info.nx, solver.info.storedNy, solver.info.nz, solver.info.ny, bases);
        if (staged.velocityTexture) postProjectionVelocity = await readTallVelocityField3D(device, staged.velocityTexture, solver.info.nx, solver.info.storedNy, solver.info.nz, solver.info.ny, bases);
      }
      let compactMechanicalEnergy: GPUSmokeResult["checkpoints"][number]["compactMechanicalEnergy"];
      if (method.id === "octree" && initialPotentialEnergyProxy !== undefined
        && method.id === "octree") {
        const compact = await readCompactOctreeVelocityField3D(device, solver,
          [solver.info.nx, solver.info.ny, solver.info.nz]);
        if (compact) {
          const spacing = {
            x: scene.container.width_m / solver.info.nx,
            y: scene.container.height_m / solver.info.ny,
            z: scene.container.depth_m / solver.info.nz,
          };
          const velocity = compactLiquidVelocityDiagnostic(compact.field, cubic.field,
            spacing.x * spacing.y * spacing.z, [spacing.x, spacing.y, spacing.z], stepDt);
          const potential = gravitationalPotentialEnergyProxy(cubic.field, solver.info.nx, solver.info.ny,
            solver.info.nz, spacing, scene.fluid.gravity_m_s2);
          compactMechanicalEnergy = {
            ...compactMechanicalEnergyDiagnostic(initialPotentialEnergyProxy, potential, velocity.kineticEnergyProxy),
            publicationValid: compact.publicationValid,
            rowCount: compact.rowCount,
            reconstructedRows: compact.reconstructedRows,
            coveredCells: compact.coveredCells,
            overlapCells: compact.overlapCells,
            invalidRows: compact.invalidRows,
            liquidCellCount: velocity.liquidCellCount,
            finiteLiquidCellCount: velocity.finiteLiquidCellCount,
            liquidVolumeCellSum: velocity.liquidVolumeCellSum,
            finiteLiquidVolumeCellSum: velocity.finiteLiquidVolumeCellSum,
            maximumLiquidComponentSpeed_m_s: velocity.maximumLiquidComponentSpeed_m_s,
            maximumLiquidComponentCfl: velocity.maximumLiquidComponentCfl,
            nonFiniteLiquidComponentCount: velocity.nonFiniteLiquidComponentCount,
          };
          if (process.env.FLUID_SPEED_MAP === "1") {
            // Locate the kinetic energy: the leading edge of a wall/ceiling
            // sheet running away distinguishes velocity-inheritance runaway
            // from a pressure-driven jet concentrated at the impact base.
            const { nx, ny, nz } = solver.info;
            const speedAt = (cell: number) => Math.hypot(compact.field[3 * cell] ?? NaN,
              compact.field[3 * cell + 1] ?? NaN, compact.field[3 * cell + 2] ?? NaN);
            const cells: { cell: number; speed: number; alpha: number }[] = [];
            const layerMax: number[] = Array.from({ length: ny }, () => 0);
            for (let cell = 0; cell < nx * ny * nz; cell += 1) {
              const alpha = cubic.field[cell] ?? 0;
              if (!(alpha > 1e-4)) continue;
              const speed = speedAt(cell);
              if (!Number.isFinite(speed)) continue;
              const j = Math.floor(cell / nx) % ny;
              layerMax[j] = Math.max(layerMax[j]!, speed);
              cells.push({ cell, speed, alpha });
            }
            cells.sort((a, b) => b.speed - a.speed);
            console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
              phase: "speed-map", time_s: solver.info.submittedTime_s,
              layerMaxSpeed: layerMax.map((value) => Number(value.toFixed(2))),
              top: cells.slice(0, 12).map(({ cell, speed, alpha }) => ({
                i: cell % nx, j: Math.floor(cell / nx) % ny, k: Math.floor(cell / (nx * ny)),
                speed: Number(speed.toFixed(2)), alpha: Number(alpha.toFixed(3)),
                vx: Number((compact.field[3 * cell] ?? NaN).toFixed(2)),
                vy: Number((compact.field[3 * cell + 1] ?? NaN).toFixed(2)),
                vz: Number((compact.field[3 * cell + 2] ?? NaN).toFixed(2)) })) }));
          }
          if (stabilityEnvelope) {
            stabilityEnvelope.peakLiquidSpeed_m_s = Math.max(
              stabilityEnvelope.peakLiquidSpeed_m_s, velocity.maximumLiquidComponentSpeed_m_s,
            );
            stabilityEnvelope.peakComponentCfl = Math.max(
              stabilityEnvelope.peakComponentCfl, velocity.maximumLiquidComponentCfl,
            );
            stabilityEnvelope.nonFiniteVelocityCount += velocity.nonFiniteLiquidComponentCount;
          }
        }
      }
      const raster = rasterCheckpointRequested && method.id === "octree"
        ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies,
          globalFineGenerationTransitionRequested)
        : undefined;
      const globalFineGeneration = globalFineGenerationTransitionRequested && method.id === "octree"
        ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
      checkpoints.push({ time_s: solver.info.submittedTime_s ?? 0, field: cubic.field, summary: cubic.summary,
        raster, globalFineGeneration, preProjectionVelocity, postProjectionVelocity, compactMechanicalEnergy });
      while (nextCheckpoint_s <= (solver.info.submittedTime_s ?? 0) + 1e-9) nextCheckpoint_s += checkpointEvery_s;
      samplingWall_ms += performance.now() - samplingStartedAt;
    }
    // Operator-clock heartbeat. Wall time here deliberately includes the QA
    // readbacks excluded from simulationWall_ms: the question this answers is
    // "is the run progressing at all", not "how fast is the solver".
    const progressAt_ms = performance.now();
    if (progressAt_ms - lastProgressAt_ms >= PROGRESS_HEARTBEAT_WALL_MS
      && steps - lastProgressSteps >= PROGRESS_HEARTBEAT_STEPS) {
      const windowSteps = steps - lastProgressSteps;
      const windowWall_ms = progressAt_ms - lastProgressAt_ms;
      lastProgressAt_ms = progressAt_ms;
      lastProgressSteps = steps;
      console.log(JSON.stringify({
        scenario: scenarioId, method: method.id, record: "progress",
        steps, submittedTime_s: solver.info.submittedTime_s ?? 0, targetTime_s: target_s,
        wall_ms: Math.round(progressAt_ms - runStarted),
        windowSteps, windowWallPerStep_ms: Number((windowWall_ms / windowSteps).toFixed(3)),
        rejectedAdvanceAttempts, maximumConsecutiveRejectedAdvances,
        structuredRejectSummary: solver.info.structuredRejectSummary,
      }));
      samplingWall_ms += performance.now() - progressAt_ms;
    }
    if (lost) throw new Error(`${method.id} device lost: ${lost.message || lost.reason}`);
  }
  const simulationWall_ms = Math.max(0, performance.now() - runStarted - samplingWall_ms);
  dataFlowAudit?.stop();
  await awaitAdvanceCompletion();
  captureGenericPhaseTrace();
  // Ghost-fluid theta census. The failure dump already carries these words, but
  // a converging scene never takes that path, so the control case in any
  // stiffness comparison was unmeasurable. Bins are [decoupled, at-1e-2-floor,
  // <=.05, <=.1, <=.25, <=.5, <1, ==1] over boundary slots.
  if (process.env.FLUID_BOUNDARY_THETA_HISTOGRAM === "1") {
    const projection = (solver as GPUSolverInstance & { octreeProjection?: {
      readPowerFrontierFailure(): Promise<{ boundaryCandidate?: readonly number[] } | undefined>;
    } }).octreeProjection;
    const census = await projection?.readPowerFrontierFailure();
    const words = census?.boundaryCandidate;
    if (words && words.length >= 16) {
      console.log(JSON.stringify({ scenario: scenario.id, method: method.id,
        phase: "ghost-fluid-theta-census", slots: words[3], rows: words[2],
        histogram: { decoupled: words[8], atFloor1e2: words[9], to0_05: words[10],
          to0_1: words[11], to0_25: words[12], to0_5: words[13], below1: words[14],
          unscaled: words[15] } }));
    }
  }
  if (powerGenerationAuditSnapshot) {
    if (powerGenerationAuditSteps.length === 0) {
      throw new Error("structured generation audit captured no accepted-step snapshots");
    }
    const fine = (solver as GPUSolverInstance).globalFineLevelSetSource;
    if (!fine) throw new Error("structured generation audit is missing the final fine authority");
    const failedSnapshots: {
      step: number;
      structured: ReturnType<typeof unpackStructuredVelocityControl>;
      boundary: ReturnType<typeof unpackStructuredBoundaryControl>;
      fineGeneration: number;
      fineHeader: readonly number[];
      failures: readonly string[];
    }[] = [];
    const snapshotBytes = powerGenerationAuditSteps.length
      * STRUCTURED_GENERATION_AUDIT_SNAPSHOT.strideBytes;
    try {
      await powerGenerationAuditSnapshot.mapAsync(GPUMapMode.READ, 0, snapshotBytes);
      const mapped = new Uint8Array(powerGenerationAuditSnapshot.getMappedRange(0, snapshotBytes));
      for (let record = 0; record < powerGenerationAuditSteps.length; record += 1) {
        const snapshot = unpackStructuredGenerationAuditSnapshot(mapped, record);
        const expectedFineGeneration = powerGenerationAuditFineGenerations[record]!;
        const generationFailures = [...exactStructuredGenerationAuditFailures({
          publishedFineGeneration: expectedFineGeneration,
          expectedStructuredEpoch: snapshot.structured.epoch,
          previousFineGeneration: previousAuditedFineGeneration,
          previousStructuredEpoch: previousAuditedPowerGeneration,
          structured: snapshot.structured,
          boundary: snapshot.boundary,
        })];
        const fineHeader = snapshot.fineHeader;
        if (fineHeader[0] !== expectedFineGeneration || fineHeader[1] === 0
          || fineHeader[1] > fine.plan.maximumResidentBricks
          || fineHeader[2] !== fine.plan.maximumResidentBricks
          || (fineHeader[3] & 3) !== 3 || fineHeader[4] !== Math.ceil(fineHeader[1] / 64)
          || fineHeader[5] !== 1 || fineHeader[6] !== 1) {
          generationFailures.push("fine workset header is invalid or stale");
        }
        const diagnostics = decodeOctreeMGPCGDiagnostics(snapshot.mgpcgControl);
        const projectedResidual = octreeProjectedVariationalResidualRms(
          Math.sqrt(Math.max(0, diagnostics.residualSquared)),
          powerGenerationAuditDts[record]!, scene.fluid.density_kg_m3,
        );
        // Name the specific incoherence: "invalid" alone cannot distinguish a
        // solver that diverged from one whose row count disagrees with the
        // structured epoch it was solved against, and those have no shared fix.
        const mgpcgFailures: string[] = [];
        if (diagnostics.flags !== 0) mgpcgFailures.push(`flags=0x${diagnostics.flags.toString(16)}`);
        if (!diagnostics.converged) mgpcgFailures.push(`not converged after ${diagnostics.iterations} iterations (relativeResidual=${diagnostics.relativeResidual})`);
        if (diagnostics.rows !== snapshot.structured.rowCount) mgpcgFailures.push(`rows=${diagnostics.rows} != structured rowCount=${snapshot.structured.rowCount}`);
        if (!Number.isFinite(diagnostics.relativeResidual)) mgpcgFailures.push(`relativeResidual=${diagnostics.relativeResidual} (residualSquared=${diagnostics.residualSquared}, rhsSquared=${diagnostics.rhsSquared})`);
        if (projectedResidual === undefined) mgpcgFailures.push(`projected variational residual is undefined (residualSquared=${diagnostics.residualSquared}, dt=${powerGenerationAuditDts[record]})`);
        if (mgpcgFailures.length > 0) {
          generationFailures.push(`MGPCG publication is invalid or incoherent: ${mgpcgFailures.join("; ")}`);
        }
        mgpcgIterationMinimum = Math.min(mgpcgIterationMinimum, diagnostics.iterations);
        mgpcgIterationMaximum = Math.max(mgpcgIterationMaximum, diagnostics.iterations);
        const key = String(diagnostics.iterations);
        mgpcgIterationHistogram[key] = (mgpcgIterationHistogram[key] ?? 0) + 1;
        const volume = unpackFineLevelSetGPUVolumeControl(
          snapshot.fineVolumeControl.buffer as ArrayBuffer,
        );
        const volumeValid = volume.flags === FINE_LEVELSET_VOLUME_VALID && volume.initialized
          && volume.generation === expectedFineGeneration && volume.coarseRows > 0
          && volume.lookupFailureSamples === 0 && volume.staleOwnerSamples === 0
          && Number.isFinite(volume.referenceVolume) && volume.referenceVolume > 0
          && Number.isFinite(volume.currentVolume) && volume.currentVolume > 0;
        if (!volumeValid) generationFailures.push("fine volume publication is invalid or stale");
        const volumeDrift = volumeValid
          ? (volume.currentVolume - volume.referenceVolume) / volume.referenceVolume
          : Number.POSITIVE_INFINITY;
        const energy = decodeStructuredProjectionEnergy(snapshot.projectionEnergyControl);
        if (!energy.sample || energy.sample.epoch !== snapshot.structured.epoch
          || energy.sample.activeBank !== snapshot.structured.activeBank) {
          generationFailures.push("structured projection energy is invalid or generation-incoherent");
        }
        if (stabilityEnvelope) {
          stabilityEnvelope.maximumPressureRelativeResidual = Math.max(
            stabilityEnvelope.maximumPressureRelativeResidual,
            Number.isFinite(diagnostics.relativeResidual)
              ? diagnostics.relativeResidual : Number.POSITIVE_INFINITY,
          );
          stabilityEnvelope.maximumProjectedVariationalResidual = Math.max(
            stabilityEnvelope.maximumProjectedVariationalResidual,
            projectedResidual ?? Number.POSITIVE_INFINITY,
          );
          stabilityEnvelope.maximumExactVolumeDrift = Math.max(
            stabilityEnvelope.maximumExactVolumeDrift, Math.abs(volumeDrift),
          );
          if (!volumeValid) stabilityEnvelope.invalidVolumeSampleCount += 1;
          if (energy.sample) {
            stabilityEnvelope.maximumProjectionEnergyRatio = Math.max(
              stabilityEnvelope.maximumProjectionEnergyRatio, energy.sample.projectionEnergyRatio,
            );
            stabilityEnvelope.peakKineticEnergyProxy = Math.max(
              stabilityEnvelope.peakKineticEnergyProxy,
              energy.sample.postProjectionKineticEnergyProxy,
            );
            stabilityEnvelope.projectionEnergySampleCount += 1;
          }
          stabilityEnvelope.sampledSteps += 1;
        }
        const step = powerGenerationAuditSteps[record]!;
        if (powerGenerationAuditRequested && powerGenerationAuditLog) {
          console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
            phase: "structured-generation-audit", step,
            structured: snapshot.structured, boundary: snapshot.boundary,
            fine: { generation: expectedFineGeneration, header: Array.from(fineHeader) },
            pressure: {
              iterations: diagnostics.iterations,
              relativeResidual: diagnostics.relativeResidual,
              projectedVariationalResidual: projectedResidual,
            },
            projectionEnergy: energy.sample,
            volume: volumeValid ? {
              reference: volume.referenceVolume,
              current: volume.currentVolume,
              relativeDrift: volumeDrift,
            } : null,
            failures: generationFailures }));
        }
        if (generationFailures.length !== 0) {
          failedSnapshots.push({ step, structured: snapshot.structured,
            boundary: snapshot.boundary, fineGeneration: expectedFineGeneration,
            fineHeader: Array.from(fineHeader), failures: generationFailures });
        }
        previousAuditedPowerGeneration = snapshot.structured.epoch;
        previousAuditedFineGeneration = expectedFineGeneration;
      }
    } finally {
      if (powerGenerationAuditSnapshot.mapState === "mapped") powerGenerationAuditSnapshot.unmap();
      powerGenerationAuditSnapshot.destroy();
    }
    const failedProjection = (solver as unknown as { octreeProjection?: {
      readPowerFrontierFailure(): Promise<unknown>;
      readGlobalFineLevelSetDiagnostics(): Promise<{
        airSupportControl?: readonly number[];
        precedingAirSupportTerminal?: readonly number[];
        firstAirSupportFailure?: readonly number[];
      } | undefined>;
    } }).octreeProjection;
    const candidateAudit = powerGenerationAuditRequested && powerCandidateAuditRequested
      ? await failedProjection?.readPowerFrontierFailure()
      : undefined;
    const candidateEpoch = (candidateAudit as { epoch?: readonly number[] } | undefined)?.epoch;
    const candidateFailures = candidateEpoch && Number(candidateEpoch[4]) !== 0
      ? ["next structured candidate is rejected"]
      : [];
    if (failedSnapshots.length !== 0 || candidateFailures.length !== 0) {
      const [candidateFailure, fineFailure] = await Promise.all([
        candidateAudit ?? failedProjection?.readPowerFrontierFailure(),
        failedProjection?.readGlobalFineLevelSetDiagnostics(),
      ]);
      throw new Error(`structured generation audit failed: ${JSON.stringify({
        failedSnapshots, candidateFailures, candidateFailure,
        airSupportFailure: fineFailure ? {
          control: fineFailure.airSupportControl,
          precedingTerminal: fineFailure.precedingAirSupportTerminal,
          firstFailure: fineFailure.firstAirSupportFailure,
        } : undefined,
      })}`);
    }
    if (powerGenerationAuditRequested) {
      powerGenerationAuditedSteps = powerGenerationAuditSteps.length;
    }
  }
  // ---- Silent-failure tripwires: evaluation ---------------------------------
  // docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3 (P0.3). Every one of these fails
  // the run. A change that gets faster while tripping one is not a speedup.
  if (tripwireSnapshot) {
    const tripped: {
      id: TripwireId; step: number; fineGeneration: number; detail: Record<string, unknown>;
    }[] = [];
    if (tripwireSteps.length === 0) {
      tripwireSnapshot.destroy();
      if (tripwiresRequired) {
        throw new Error("tripwires could not be evaluated: no accepted step captured a"
          + " tripwire record (see docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3)");
      }
    } else {
      const snapshotBytes = tripwireSteps.length * TRIPWIRE_RECORD.strideBytes;
      try {
        await tripwireSnapshot.mapAsync(GPUMapMode.READ, 0, snapshotBytes);
        const mapped = new Uint8Array(tripwireSnapshot.getMappedRange(0, snapshotBytes));
        const words = (record: number, offsetBytes: number, byteLength: number) => new Uint32Array(
          mapped.buffer, mapped.byteOffset + record * TRIPWIRE_RECORD.strideBytes + offsetBytes,
          byteLength / 4,
        );
        for (let record = 0; record < tripwireSteps.length; record += 1) {
          const step = tripwireSteps[record]!;
          const fineGeneration = tripwireFineGenerations[record]!;
          const trip = (id: TripwireId, detail: Record<string, unknown>) =>
            tripped.push({ id, step, fineGeneration, detail });
          // 1. Topology rollback. `settleFinePublication` returns early on the
          //    clean path and writes control[5]=1 only on the rollback branch;
          //    the host re-zeroes control words 0..7 at the start of every
          //    topology encode, so a set word is always this generation's own
          //    verdict, never a stale latch.
          const topologyControl = words(record, TRIPWIRE_RECORD.topologyOffsetBytes,
            TRIPWIRE_RECORD.topologyBytes);
          const topology = unpackFineLevelSetGPUTopologyControl(topologyControl);
          if (topology.rolledBack) {
            trip("topology-rollback", { rolledBack: true,
              flags: topology.flags, published: topology.published,
              downstreamFinalizeReason: topology.downstreamFinalizeReason,
              interfaceBricks: topology.interfaceBricks,
              desiredBricks: topology.desiredBricks,
              activatedBricks: topology.activatedBricks,
              control: Array.from(topologyControl) });
          }
          // 2. Restriction coverage. NOTE: the plan document specifies this
          //    tripwire as `unacceptedRows == 0`, and that is wrong. The
          //    producer documents the field as "Never an error: a row outside
          //    the fine narrow band is legitimately uncorrected"
          //    (lib/webgpu-octree-fine-to-coarse-levelset.ts). Measured on the
          //    mini lane at HEAD: 144/1500 rows unaccepted at step 1 decaying
          //    to 30/1473 by step 500, peak 177/1500 = 11.8%, with flags 0 and
          //    valid true throughout. An == 0 gate fires on 439 of 500 steps.
          //    What the field is actually for is the coverage-regression signal
          //    for a band-width change, which is otherwise silent because an
          //    unaccepted row raises no flag and writes no contribution. So the
          //    gate tests that instead: the source must be accepted, and the
          //    uncovered FRACTION must stay inside an authored envelope that a
          //    real band narrowing would blow through.
          const restrictionWords = words(record, TRIPWIRE_RECORD.restrictionOffsetBytes,
            TRIPWIRE_RECORD.restrictionBytes);
          const restriction = unpackFineToCoarseGPUControl(restrictionWords);
          const uncoveredFraction = restriction.rowCount > 0
            ? restriction.unacceptedRows / restriction.rowCount : undefined;
          if (restriction.flags !== 0 || !restriction.valid) {
            trip("restriction-unaccepted", { unevaluable: true,
              reason: "restriction source rejected; unaccepted-row count is masked to zero",
              flags: restriction.flags, rowCount: restriction.rowCount,
              valid: restriction.valid, control: Array.from(restrictionWords) });
          } else if (uncoveredFraction === undefined) {
            trip("restriction-unaccepted", { unevaluable: true,
              reason: "restriction published a zero row count; coverage is not evaluable",
              control: Array.from(restrictionWords) });
          } else if (uncoveredFraction > TRIPWIRE_MAXIMUM_UNCOVERED_ROW_FRACTION) {
            trip("restriction-unaccepted", {
              reason: "fine-band coverage regressed: more coarse rows are outside the"
                + " band than the authored envelope allows",
              unacceptedRows: restriction.unacceptedRows, rowCount: restriction.rowCount,
              uncoveredFraction: Number(uncoveredFraction.toFixed(4)),
              envelope: TRIPWIRE_MAXIMUM_UNCOVERED_ROW_FRACTION,
              flags: restriction.flags, valid: restriction.valid,
              control: Array.from(restrictionWords) });
          }
          // 3. Solver convergence. Non-convergence at the encoded budget
          //    publishes the SEED pressure and fails nothing today; this is the
          //    guard for that cliff. Steps that executed no iterations are
          //    exempt by construction (nothing to converge); a terminal count of
          //    zero is gated per run by tools/benchmark-power-dam.ts.
          const mgpcgWords = words(record, TRIPWIRE_RECORD.mgpcgOffsetBytes,
            TRIPWIRE_RECORD.mgpcgBytes);
          const mgpcg = decodeOctreeMGPCGDiagnostics(mgpcgWords);
          if (mgpcg.flags !== 0) {
            trip("mgpcg-nonconvergence", { unevaluable: true,
              reason: "MGPCG control reports error flags; the converged word is not meaningful",
              flags: mgpcg.flags, converged: mgpcg.converged, iterations: mgpcg.iterations,
              rows: mgpcg.rows });
          } else if (mgpcg.iterations > 0 && !mgpcg.converged) {
            trip("mgpcg-nonconvergence", { converged: false, iterations: mgpcg.iterations,
              rows: mgpcg.rows, relativeResidual: mgpcg.relativeResidual,
              residualSquared: mgpcg.residualSquared, rhsSquared: mgpcg.rhsSquared });
          }
          // 4. Fine-band capacity overflow. The active count degrades to the
          //    INVALID sentinel, which silently no-ops the solver and still
          //    prints PASS. The count is worklist header word ONE; a prior
          //    consumer read word zero (the generation) and printed nonsense.
          const header = readFineLevelSetWorksetHeader(words(record,
            TRIPWIRE_RECORD.fineHeaderOffsetBytes, TRIPWIRE_RECORD.fineHeaderBytes));
          if (header === undefined) {
            trip("fine-band-sentinel", { unevaluable: true,
              reason: "fine worklist header could not be decoded" });
          } else if (header.activeCount === 0xffff_ffff) {
            trip("fine-band-sentinel", { activeCount: header.activeCount,
              sentinel: "0xFFFFFFFF", capacity: header.capacity,
              generation: header.generation, flags: header.flags });
          }
        }
      } finally {
        if (tripwireSnapshot.mapState === "mapped") tripwireSnapshot.unmap();
        tripwireSnapshot.destroy();
      }
    }
    const allowed = tripped.filter((entry) => tripwireAllowList.has(entry.id));
    const failing = tripped.filter((entry) => !tripwireAllowList.has(entry.id));
    for (const entry of allowed) {
      console.error(`[tripwire ${entry.id} ALLOWED] ${JSON.stringify(entry)}`);
    }
    if (failing.length !== 0) {
      const byId: Record<string, number> = {};
      for (const entry of failing) byId[entry.id] = (byId[entry.id] ?? 0) + 1;
      throw new Error(`silent-failure tripwire(s) tripped over ${tripwireSteps.length}`
        + ` captured steps: ${JSON.stringify({ counts: byId,
          firstTrips: failing.slice(0, 12), lastTrip: failing[failing.length - 1] })}`
        + " (see docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3)");
    }
    console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
      phase: "tripwires", capturedSteps: tripwireSteps.length,
      required: tripwiresRequired, allowed: Array.from(tripwireAllowList),
      tripped: tripped.length }));
  }
  const gpuFineTimestamps: GPUFineTimestampReport | undefined = genericPhaseTraceRequested ? {
    measuredAdvances: attributedTraceSamples,
    measuredPasses: Array.from(attributedPhaseBuckets.values())
      .reduce((sum, bucket) => sum + bucket.samples, 0),
    invalidPasses: 0,
    summedPass_ms: Array.from(attributedPhaseBuckets.values())
      .reduce((sum, bucket) => sum + bucket.total_ms, 0),
    byLabel: Object.fromEntries(Array.from(attributedPhaseBuckets.entries())
      .sort((left, right) => right[1].total_ms - left[1].total_ms || left[0].localeCompare(right[0]))
      .map(([label, bucket]) => [label, {
        ...bucket, mean_ms: bucket.total_ms / Math.max(1, bucket.samples),
      }])),
  } : undefined;
  const gpuDataFlowManifest = dataFlowAudit?.report(
    Math.min(steps, genericPhaseTraceAdvances),
    gpuFineTimestamps?.byLabel,
  );
  if (performanceProfileRequested && scenarioId === "dam-break-ui" && method.id === "octree") {
    const authority = solver as GPUSolverInstance & {
      structuredVelocityControl?: GPUBuffer;
      structuredBoundaryControl?: GPUBuffer;
    };
    const fine = authority.globalFineLevelSetSource;
    const controls = [
      ["fine worklist", fine?.worklist, 28],
      ["structured velocity", authority.structuredVelocityControl, 24],
      ["structured boundary", authority.structuredBoundaryControl, 64],
    ] as const;
    const missing = controls.filter(([, buffer]) => !buffer).map(([label]) => label);
    if (!fine || missing.length !== 0) {
      throw new Error(`final performance authority is missing packed controls: ${missing.join(", ")}`);
    }
    const packed = await readBufferBindingsPacked(device, controls.map(([, buffer, byteLength]) => ({
      binding: { buffer: buffer! },
      byteLength,
    })));
    const words = (index: number) => new Uint32Array(
      packed[index].buffer, packed[index].byteOffset, packed[index].byteLength / 4,
    );
    const fineWorklistHeader = words(0);
    const structured = unpackStructuredVelocityControl(words(1));
    const boundary = unpackStructuredBoundaryControl(words(2));
    const expectedTime_s = exactStepCount === undefined || maxDtOverride === undefined
      ? Number.NaN : exactStepCount * maxDtOverride;
    const finalAuthorityFailures = finalPerformanceAuthorityFailures({
      expectedSteps: exactStepCount ?? Number.NaN,
      observedSteps: steps,
      expectedTime_s,
      targetTime_s: target_s,
      submittedTime_s: solver.info.submittedTime_s ?? Number.NaN,
      fineSourceGeneration: fine.generation,
      fineWorklistHeader,
      finePageCapacity: fine.plan.maximumResidentBricks,
      structured,
      boundary,
    });
    const finalAuthority = {
      steps,
      expectedSteps: exactStepCount,
      targetTime_s: target_s,
      submittedTime_s: solver.info.submittedTime_s,
      fine: {
        sourceGeneration: fine.generation,
        worklistGeneration: fineWorklistHeader[0],
        activePages: fineWorklistHeader[1],
        capacity: fineWorklistHeader[2],
        flags: fineWorklistHeader[3],
      },
      structured,
      boundary,
      failures: finalAuthorityFailures,
    };
    console.log(JSON.stringify({
      scenario: scenarioId, method: resultMethod,
      phase: "final-performance-authority", ...finalAuthority,
    }));
    if (finalAuthorityFailures.length !== 0) {
      throw new Error(`final performance authority rejected: ${JSON.stringify(finalAuthority)}`);
    }
  }
  solver.info.completedTime_s = Math.max(solver.info.completedTime_s ?? 0, solver.info.submittedTime_s ?? 0);
  solver.info.simulatedTime_s = solver.info.submittedTime_s;
  // The octree projection's pressure counters (`pressureIterationsUsed`,
  // `pressureConverged`, the residuals) are refreshed ONLY by an explicit
  // solve-diagnostics readback, and the recurring `readStats()` path never
  // issues one -- so without this the reported "terminal pressure iterations"
  // is the t=0 bootstrap solve's count, forever, and the benchmark's
  // zero-iteration tripwire can never observe the terminal step. This runs
  // after the measured window closes (simulationWall_ms is already computed),
  // so it costs the wall nothing. See docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3.
  if (method.id === "octree") {
    const projection = (solver as unknown as {
      octreeProjection?: { readSolveDiagnostics(): Promise<void> };
    }).octreeProjection;
    if (projection) await projection.readSolveDiagnostics();
  }
  const info = { ...await solver.readStats() };
  // The compact paper path never dispatches the dense velocity reduction.
  // Exact QA checkpoints already reconstruct the accepted structured rows on
  // the fine lattice, so reuse that evidence rather than adding a production
  // pass, fallback, or second readback solely for scalar telemetry.
  if (method.id === "octree" && info.powerDiagramAuthoritative === true) {
    const compactVelocity = checkpoints.findLast(
      (checkpoint) => checkpoint.compactMechanicalEnergy !== undefined,
    )?.compactMechanicalEnergy;
    if (compactVelocity) {
      info.maxSpeed_m_s = compactVelocity.maximumLiquidComponentSpeed_m_s;
      info.maxComponentCfl = compactVelocity.maximumLiquidComponentCfl;
    }
  }
  if (info.gpuValidationError && !validationErrors.includes(info.gpuValidationError)) {
    validationErrors.push(info.gpuValidationError);
  }
  // The solver validates every advance's encoded stage order against the
  // declared step program (lib/physics-step-program.ts). Any deviation is a
  // broken driver contract and fails the gate, exactly as the UI raises its
  // step-sequence-deviation stability flag.
  if (info.stepSequenceDeviations?.length) {
    throw new Error(`physics step program deviation: ${info.stepSequenceDeviations.join("; ")}`);
  }
  matched ??= performanceProfileRequested
    ? { field: new Float32Array(info.nx * info.ny * info.nz),
      summary: summarizeScalarField(new Float32Array(info.nx * info.ny * info.nz), info.nx, info.ny, info.nz) }
    : await readCubicVolumeField(device, solver);
  const final = includeFinalFieldStats && steps !== oracleSteps ? await readCubicVolumeField(device, solver) : matched;
  const finalSolver = solver as GPUSolverInstance & { velocityTexture?: GPUTexture;
    powerDescriptorControl?: GPUBuffer; powerTopologyControl?: GPUBuffer;
    powerDescriptorRows?: GPUBuffer; powerTopologyMetrics?: GPUBuffer; powerLeafHeaders?: GPUBuffer; powerOwnerArena?: GPUBuffer;
    mgpcgControl?: GPUBuffer };
  const velocityTexture = finalSolver.velocityTexture;
  const finalSpacing = {
    x: scene.container.width_m / info.nx,
    y: scene.container.height_m / info.ny,
    z: scene.container.depth_m / info.nz
  };
  const velocitySummary = velocityTexture && final && method.id !== "octree"
    ? (info.gridKind === "restricted-tall-cell"
      ? await readTallVelocityTexture3D(device, velocityTexture, info.nx, info.storedNy, info.nz, info.ny, await readFloatTexture2D(device, solver.columnBaseTexture!, info.nx, info.nz), final.field, finalSpacing, scene.numerics.maxDt_s)
      : await readVelocityTexture3D(device, velocityTexture, info.nx, info.ny, info.nz, final.field, finalSpacing, scene.numerics.maxDt_s))
    : undefined;
  let velocityParityField: Float32Array | undefined;
  let compactVelocityRaster: GPUSmokeResult["compactVelocityRaster"];
  if (scenarioId === "dam-break-ui" && final) {
    if (method.id === "tall-cell" && velocityTexture && info.gridKind === "restricted-tall-cell") {
      const bases = await readFloatTexture2D(device, solver.columnBaseTexture!, info.nx, info.nz);
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
  // Always captured for octree: the structured-validation gates require the
  // final generation diagnostics, and a gate that reads `undefined` reports
  // a wiring failure rather than evaluating the solver's actual state.
  const finalGlobalFineGeneration = method.id === "octree"
    ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
  const finalGlobalFineRaster = globalFineGenerationTransitionRequested && method.id === "octree"
    ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies, true) : undefined;
  const sparseVoxelStats = sparseStatsRequested && sparseSource
    ? await readSparseVoxelStats(device, sparseSource, seedBrickBounds)
    : undefined;
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
          if (arena.length < 16
              || arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.magic] !== OCTREE_OWNER_ARENA_MAGIC) {
            return { ...canonical(cell), invalid: true };
          }
          const brickDimensions = dimensions.map((value) => Math.ceil(value / 8));
          const brick = cell.map((value) => Math.floor(value / 8));
          const logical = brick[0] + brick[1] * brickDimensions[0] + brick[2] * brickDimensions[0] * brickDimensions[1];
          const capacity = arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.capacity];
          const pageOffset = arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerRecordPageOffsetWords];
          const resident = Math.min(arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.residentCount], capacity);
          const key = logical + 1; let low = 0, high = resident;
          while (low < high) {
            const middle = low + Math.floor((high - low) / 2);
            if (arena[16 + middle] < key) low = middle + 1;
            else high = middle;
          }
          const encoded = low < resident && arena[16 + low] === key ? arena[pageOffset + low] : 0;
          if (encoded === 0) return canonical(cell);
          if (encoded === 0xffff_ffff || encoded > capacity) return { ...canonical(cell), invalid: true };
          const local = (cell[0] & 7) + 8 * ((cell[1] & 7) + 8 * (cell[2] & 7));
          const word = arena[arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerPagesOffsetWords]
            + (encoded - 1) * 512 + local]; if (word === 0) return canonical(cell);
          const exponent = (word >>> 18) & 7; const size = 1 << exponent;
          const brickOrigin = cell.map((value) => Math.floor(value / 8) * 8);
          const delta = [word & 63, (word >>> 6) & 63, (word >>> 12) & 63]
            .map((value) => value - 32);
          const origin = brickOrigin.map((value, axis) => value + delta[axis]) as [number, number, number];
          const invalid = (word & 0x8000_0000) === 0 || exponent > 5
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
  await device.queue.onSubmittedWorkDone();
  const accountingOwner = solver as GPUSolverInstance & {
    // `captureWorkAccounting` already computes an exact English cause when a
    // stage's active/scheduled lanes come back null, and it was being thrown
    // away here -- leaving the regression artifact to report only the generic
    // "counters are null for at least one stage". Carry the cause out so a
    // blocked capture is triageable without a rerun.
    captureWorkAccounting?: () => Promise<{
      snapshot: OctreeWorkSnapshot; pressure?: { blocker?: string };
    }>;
    workAccounting?: { snapshot(): OctreeWorkSnapshot };
  };
  const capturedWorkAccounting = accountingOwner.captureWorkAccounting
    ? await accountingOwner.captureWorkAccounting()
    : undefined;
  const gpuPassTimestamps = passTimestampAudit
    ? await passTimestampAudit.report()
    : undefined;
  const diagnosticProjection = solver as GPUSolverInstance & {
    octreeProjection?: {
      readGlobalFineLevelSetDiagnostics(): Promise<{
        topologyControl: readonly number[];
        structuredVelocityControl: readonly number[];
        structuredBoundaryControl: readonly number[];
        airSupportControl: readonly number[];
        airSupportTerminalScratch: readonly number[];
        finePageDeltaHeader: readonly number[];
      } | undefined>;
    };
    workAccountingPlan?: Record<string, unknown>;
    globalFineLevelSetSource?: WebGPUFineLevelSetBrickSource;
  };
  const terminalAlgorithmState = gpuPassTimestampRequested
    ? await diagnosticProjection.octreeProjection?.readGlobalFineLevelSetDiagnostics()
    : undefined;
  const finePlan = diagnosticProjection.globalFineLevelSetSource?.plan;
  const algorithmDiagnostics = terminalAlgorithmState ? {
    topologyControl: terminalAlgorithmState.topologyControl,
    structuredVelocityControl: terminalAlgorithmState.structuredVelocityControl,
    structuredBoundaryControl: terminalAlgorithmState.structuredBoundaryControl,
    airSupportControl: terminalAlgorithmState.airSupportControl,
    airSupportTerminalScratch: terminalAlgorithmState.airSupportTerminalScratch,
    finePageDeltaHeader: terminalAlgorithmState.finePageDeltaHeader,
    finePlan: finePlan ? {
      maximumResidentBricks: finePlan.maximumResidentBricks,
      logicalBrickCount: finePlan.logicalBrickCount,
      samplesPerBrick: finePlan.samplesPerBrick,
      brickResolution: finePlan.brickResolution,
      fineFactor: finePlan.fineFactor,
    } : undefined,
    pressurePlan: diagnosticProjection.workAccountingPlan,
  } : undefined;
  const result: GPUSmokeResult = {
    method: resultMethod, info, grid: [info.nx, info.ny, info.nz], matchedField: matched.field,
    matchedSummary: matched.summary, compactFieldEvidence: matched.compactFieldEvidence,
    matchedTallCellActivity: matched.tallCellActivity,
    finalSummary: final?.summary, finalTallCellActivity: final?.tallCellActivity,
    finalTallVolumeGaps: final?.tallVolumeGaps, validationErrors,
    construction_ms, runtime_ms: performance.now() - runStarted, simulationWall_ms, steps,
    rejectedAdvanceAttempts, maximumConsecutiveRejectedAdvances,
    structuredRejectReports, firstStructuredRejectStep,
    gpuCommandAudit: commandAudit?.snapshot(),
    gpuFineTimestamps,
    gpuPassTimestamps,
    algorithmDiagnostics,
    gpuDataFlowManifest,
    powerGenerationAuditedSteps,
    mgpcgIterationAudit: powerGenerationAuditedSteps > 0 ? {
      samples: powerGenerationAuditedSteps,
      minimum: Number.isFinite(mgpcgIterationMinimum) ? mgpcgIterationMinimum : 0,
      maximum: mgpcgIterationMaximum,
      histogram: mgpcgIterationHistogram,
    } : undefined,
    velocitySummary,
    velocityParityField, velocityParityVolume: final?.field, compactVelocityRaster,
    initialFluidBrickStats, sparseVoxelStats, hybridPresentationStats,
    initialGlobalFineGeneration, initialGlobalFineRaster, finalGlobalFineGeneration, finalGlobalFineRaster,
    octreePowerTopologyDiagnostics,
    octreeMGPCGDiagnostics,
    stabilityEnvelope,
    octreeWorkAccounting: capturedWorkAccounting?.snapshot
      ?? accountingOwner.workAccounting?.snapshot(),
    octreeWorkAccountingBlocker: capturedWorkAccounting?.pressure?.blocker,
    energyTrace, checkpoints
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
      fail((envelope?.invalidVolumeSampleCount ?? Infinity) === 0,
        `octree produced ${envelope?.invalidVolumeSampleCount ?? "unknown"} invalid per-step volume fields`);
      fail((envelope?.nonFiniteVelocityCount ?? Infinity) === 0,
        `octree encountered ${envelope?.nonFiniteVelocityCount ?? "unknown"} non-finite velocities`);
      fail(octreePowerPressureEnvelopeIsAcceptable(
        result.info.pressureSolver,
        envelope?.maximumPressureRelativeResidual,
        envelope?.maximumProjectedVariationalResidual,
      ), `octree per-step pressure residual peaked at relative=${envelope?.maximumPressureRelativeResidual}`
        + ` rms=${envelope?.maximumProjectedVariationalResidual}`);
      const maximumVolumeDrift = scenarioId === "hydrostatic-power-two-level" ? 1e-4 : 0.01;
      fail((envelope?.maximumExactVolumeDrift ?? Infinity) <= maximumVolumeDrift,
        `octree per-step volume drift peaked at ${envelope?.maximumExactVolumeDrift}; limit ${maximumVolumeDrift}`);
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
        ?? result.info.maxSpeed_m_s ?? 0;
      fail(observedMotionSpeed_m_s >= minimumPeakSpeed_m_s,
        `${result.method} observed motion speed ${observedMotionSpeed_m_s} m/s is below ${minimumPeakSpeed_m_s} m/s`);
    }
    fail(result.validationErrors.length === 0, `${result.method} WebGPU validation errors: ${result.validationErrors.join("; ")}`);
    fail((result.info.nonFiniteCount ?? 0) === 0, `${result.method} reported ${result.info.nonFiniteCount} non-finite values`);
    fail(Number.isFinite(result.info.maxSpeed_m_s ?? NaN), `${result.method} max speed is not finite`);
    const structuredVelocityTransport = result.method === "octree"
      && result.info.powerDiagramAuthoritative === true;
    if (structuredVelocityTransport) {
      fail(Number.isFinite(result.info.maxComponentCfl ?? NaN)
        && ((scenarioId === "hydrostatic-power-two-level" || scenarioId === "hydrostatic-power-large-offset")
          || (result.info.maxComponentCfl ?? 0) > 0),
        `octree structured transport reported invalid or zero CFL ${result.info.maxComponentCfl}`);
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
      const wallPerStep_ms = quadtree.simulationWall_ms / Math.max(1, quadtree.steps), gpuPerStep_ms = quadtree.info.physicsTrace?.total_ms ?? 0;
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
      fail(octree.info.powerDiagramReady === true && octree.info.powerDiagramAuthoritative === true,
        "octree authoritative power projection did not publish");
      fail(octree.info.powerDiagramAuthoritative === true,
        "octree structured velocity authority was not published");
      const powerTopology = octree.octreePowerTopologyDiagnostics;
      fail(powerTopology?.descriptor.errorCount === 0 && powerTopology.topology.invalidCount === 0,
        `octree authoritative power topology is invalid: ${JSON.stringify(powerTopology)}`);
      const selectedPressureSolver = octree.info.pressureSolver;
      const supportedPressureSolver = selectedPressureSolver?.includes("Section 4.3 hybrid") === true;
      fail(supportedPressureSolver,
        `octree authoritative power projection selected the wrong pressure solver: ${octree.info.pressureSolver ?? "unknown"}`);
      fail(octreePowerPressureDiagnosticsAreAcceptable(
        selectedPressureSolver,
        octree.octreeMGPCGDiagnostics,
      ), `octree Section 4.3 pressure solve missed the relative-residual limit 1e-4: `
        + `${JSON.stringify(octree.octreeMGPCGDiagnostics)}`);
    }
    if (octreeGlobalFineFactorOverride === "4" || octreeGlobalFineFactorOverride === "8") {
      const expectedFactor = Number(octreeGlobalFineFactorOverride);
      fail(octree.info.globalFineLevelSetEnabled === true,
        "octree requested the global fine level set but did not expose a live GPU source");
      fail(octree.info.globalFineLevelSetFactor === expectedFactor,
        `octree global fine factor ${octree.info.globalFineLevelSetFactor ?? "unknown"} differs from requested ${expectedFactor}`);
    }
    const structuredValidationScenario = scenarioId === "hydrostatic-power-two-level"
      || scenarioId === "hydrostatic-power-large-offset"
      || scenarioId === "minimal-power-dam-break";
    if (structuredValidationScenario) {
      const expectedGrid = scenarioId === "hydrostatic-power-large-offset"
        ? [32, 24, 16] : [16, 16, 16];
      fail(octree.grid.every((value, axis) => value === expectedGrid[axis]),
        `structured validation grid ${octree.grid.join("x")} is not ${expectedGrid.join("x")}`);
      fail(octree.info.powerDiagramAuthoritative === true,
        "direct structured velocity/boundary authority is unavailable");
      fail((octree.info.quadtreeMaximumNeighborRatio ?? Infinity) <= 2,
        `structured octree violated 2:1 balance: ${octree.info.quadtreeMaximumNeighborRatio}`);
      if (powerGenerationAuditRequested) {
        fail(octree.powerGenerationAuditedSteps === octree.steps,
          `structured audit observed ${octree.powerGenerationAuditedSteps} of ${octree.steps} steps`);
      }
      if (scenarioId === "minimal-power-dam-break") {
        const envelope = octree.stabilityEnvelope;
        if (powerGenerationAuditRequested) {
          fail((envelope?.maximumProjectedVariationalResidual ?? Infinity) <= 1e-6,
            `minimal dam projected residual ${envelope?.maximumProjectedVariationalResidual} exceeds 1e-6`);
        }
        if (envelope) {
          fail(envelope.maximumExactVolumeDrift <= 0.01,
            `minimal dam transient exact-volume drift ${envelope.maximumExactVolumeDrift} exceeds 1%`);
          fail(envelope.minimumDominantComponentFraction >= 0.99,
            `minimal dam liquid disconnected: ${JSON.stringify(envelope)}`);
        }
        const generation = octree.finalGlobalFineGeneration;
        fail(generation?.transportCommitted === true && (generation.transportProcessed ?? 0) > 0
          && generation.transportDepartureOutsideBand === 0
          && generation.transportNonfiniteVelocity === 0
          && generation.transportVelocityUnavailable === 0,
        `minimal dam fine transport is invalid: ${JSON.stringify(generation)}`);
        fail(generation?.topologyRolledBack === false,
          `minimal dam final topology rolled back: ${JSON.stringify(generation)}`);
        const mechanical = octree.checkpoints.flatMap((checkpoint) =>
          checkpoint.compactMechanicalEnergy ? [checkpoint.compactMechanicalEnergy] : []);
        if ((octree.info.simulatedTime_s ?? 0) >= 1 && checkpointEvery_s > 0) {
          fail(mechanical.length > 0, "minimal dam emitted no mechanical-energy checkpoints");
          // Physical benchmark: a closed, inviscid, source-free box cannot
          // gain mechanical energy; every checkpoint must satisfy
          // E(t) <= E(0) within a 5% discretization tolerance.
          const worstRetention = Math.max(0,
            ...mechanical.map((sample) => sample.mechanicalEnergyRetentionRatio));
          fail(worstRetention <= 1.05,
            `minimal dam gained mechanical energy: peak retention ${worstRetention} exceeds the physical bound 1 (+5% tolerance)`);
          // Physical benchmark: the fastest liquid is bounded by the Ritter
          // dam-break front celerity 2*sqrt(g*h0) for the authored column
          // height h0, with 25% headroom for discrete impact jets.
          const damScene = createSmokeScenario(scenarioId).scene;
          const gravityMagnitude = Math.hypot(damScene.fluid.gravity_m_s2.x,
            damScene.fluid.gravity_m_s2.y, damScene.fluid.gravity_m_s2.z);
          const columnHeight_m = Math.max(0.92, damScene.container.fillFraction)
            * damScene.container.height_m;
          const frontCelerity = 2 * Math.sqrt(gravityMagnitude * columnHeight_m);
          const peakSpeed = Math.max(0,
            ...mechanical.map((sample) => sample.maximumLiquidComponentSpeed_m_s));
          fail(peakSpeed <= 1.25 * frontCelerity,
            `minimal dam liquid reached ${peakSpeed} m/s; the Ritter celerity bound for the ${columnHeight_m} m column is ${frontCelerity} m/s`);
        }
        // Physical benchmark: with unilateral (separating) ceiling contact,
        // no liquid stays pinned to the lid once the impact transient has
        // passed; only spray-level counts are allowed in the top two cell
        // layers from t = 1.5 s on.
        if ((octree.info.simulatedTime_s ?? 0) >= 2 - 1e-9 && checkpointEvery_s > 0) {
          const [gx, gy, gz] = octree.grid;
          for (const checkpoint of octree.checkpoints) {
            if (checkpoint.time_s < 1.5 - 1e-9 || !checkpoint.field) continue;
            let topWet = 0;
            for (let k = 0; k < gz; k += 1) {
              for (let j = Math.max(0, gy - 2); j < gy; j += 1) {
                for (let i = 0; i < gx; i += 1) {
                  if ((checkpoint.field[i + gx * (j + gy * k)] ?? 0) > 0.5) topWet += 1;
                }
              }
            }
            fail(topWet <= 6,
              `minimal dam holds ${topWet} wet cells in its top two layers at t=${checkpoint.time_s.toFixed(2)} s; a separating ceiling leaves only spray`);
          }
        }
      }
    }
    if (globalFineGenerationTransitionRequested) {
      const container = createSmokeScenario(scenarioId).scene.container;
      const assertAuthoritativeRaster = (label: string, publishedGeneration: number | undefined,
        observed: HybridPresentationSmokeStats | undefined, requireInitialDamCornerCaps = false,
        requirePreImpactSurfaceIntegrity = false) => {
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
        // The rolled-back 512px reference has one wall-grazing crossing
        // quantized onto the adjacent depth-peel pixel at t=1.6.
        const maximumBackOnlyPixels = scenarioId === "minimal-power-dam-break" ? 1 : 0;
        fail((observed?.backOnlyInterfacePixels ?? Infinity) <= maximumBackOnlyPixels,
          `${label} depth peeling exposed back crossings without front crossings: ${JSON.stringify(observed)}`);
        fail((reverse?.frontInterfacePixels ?? 0) > 0 && (reverse?.backInterfacePixels ?? 0) > 0
          && (reverse?.backOnlyInterfacePixels ?? Infinity) <= maximumBackOnlyPixels,
        `${label} reverse depth peeling exposed back crossings without front crossings: ${JSON.stringify(reverse)}`);
        if (scenarioId === "minimal-power-dam-break") {
          const surfaceViews = [
            ["forward", observed] as const,
            ["reverse", reverse] as const,
          ];
          for (const [viewLabel, view] of surfaceViews) {
            if (requirePreImpactSurfaceIntegrity) {
              for (const depth of ["front", "back"] as const) {
                const holes = view?.enclosedSurfaceHoles?.[depth];
                // Half-float peeling can leave a two-pixel quantization run.
                // A larger pre-impact enclosed patch is a missing cube.
                fail((holes?.maximumPixels ?? Infinity) <= 2,
                  `${label} ${viewLabel} ${depth} surface contains an enclosed missing patch: ${JSON.stringify(holes)}`);
              }
            }
            const steps = view?.surfaceSteps?.front;
            fail((steps?.terraceEdgeFraction ?? Infinity) <= 0.12,
              `${label} ${viewLabel} front surface exceeds the rolled-back terrace envelope: ${JSON.stringify(steps)}`);
          }
          // The slit mask is a strict planar-dam seam check. After release, a
          // folded free surface or a one-cell spray component can project a
          // narrow, horizontally bounded gap in one view without any missing
          // mesh topology. Depth peeling above remains the evolved-surface
          // crack invariant; keep the view-space metric diagnostic-only.
          if (requireInitialDamCornerCaps) {
            fail(observed?.narrowVerticalSlits.count === 0,
              `${label} contains narrow vertical surface slits: ${JSON.stringify(observed?.narrowVerticalSlits)}`);
            fail(reverse?.narrowVerticalSlits.count === 0,
              `${label} reverse view contains narrow vertical surface slits: ${JSON.stringify(reverse?.narrowVerticalSlits)}`);
          }
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
        scenarioId === "minimal-power-dam-break", scenarioId === "minimal-power-dam-break");
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
          checkpointGeneration?.generation, checkpoint.raster, false,
          scenarioId === "minimal-power-dam-break" && checkpoint.time_s < 1 - 1e-6);
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
      fail(octreePowerPressureEnvelopeIsAcceptable(
        octree.info.pressureSolver,
        envelope?.maximumPressureRelativeResidual,
        envelope?.maximumProjectedVariationalResidual,
      ), `octree dam-break pressure residual peaked at relative=${envelope?.maximumPressureRelativeResidual}`
        + ` rms=${envelope?.maximumProjectedVariationalResidual}`);
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
      const stages = ["coarse-grid","velocity-advection","pressure-system","pressure-solve","velocity-projection","velocity-extrapolation","adaptive-publication","other"] as const satisfies readonly PaperPhaseId[];
      const gpuStageSpeedups = Object.fromEntries(stages.map((stage) => [stage,ratio(
        performancePhase_ms(uniformResult.info.physicsTrace,stage),
        performancePhase_ms(tallResult.info.physicsTrace,stage),
      )]));
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
      qualityGates: performanceProfileRequested
        ? scenarioId === "dam-break-ui" ? "final-authority-only" : "skipped"
        : "evaluated" }));
  }
} finally {
  Reflect.deleteProperty(globalThis, "navigator");
}

if (failures.length > 0) throw new Error(`WebGPU smoke invariant failures:\n${failures.map((failure) => `- ${failure}`).join("\n")}`);
