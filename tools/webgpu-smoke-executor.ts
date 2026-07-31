import { pathToFileURL } from "node:url";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import { tallCellMethod } from "../lib/methods/tall-cell";
import type { GPUSolverInstance, SimulationMethod } from "../lib/methods/types";
import { uniformMethod } from "../lib/methods/uniform";
import { quadtreeTallCellMethod } from "../lib/methods/quadtree-tall-cell";
import { octreeMethod } from "../lib/methods/octree";
import { initializeRigidBodies } from "../lib/rigid-body";
import type { SceneDescription } from "../lib/model";
import { createSingleTallCellProbeControlLayout, createSingleTallCellProbeLayout, createTallCellLayout, tallCellSettings, type SingleTallCellProbeOptions } from "../lib/tall-cell-grid";
import { WebGPUEulerianSolver, type GPUEulerianInfo, type GPUQuality } from "../lib/webgpu-eulerian";
import { summarizeDriftOscillation } from "../lib/tall-cell-diagnostics";
import { fineTopologyRetainsBackgroundOctree } from "../lib/octree-consumer-sampling";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import { FINE_LEVELSET_VOLUME_VALID, unpackFineLevelSetGPUVolumeControl }
  from "../lib/webgpu-octree-fine-levelset-volume";
import { FINE_LEVELSET_TOPOLOGY_ERROR, unpackFineLevelSetGPUTopologyControl }
  from "../lib/webgpu-octree-fine-levelset-topology";
import { unpackFineToCoarseGPUControl }
  from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { unpackOctreePowerCoarseLevelSetControl }
  from "../lib/webgpu-octree-power-coarse-levelset";
import { auditSection5FineRestriction }
  from "../lib/power-liquids-restriction-audit";
import { readFineLevelSetWorksetHeader } from "../lib/octree-fine-levelset-bricks";
import { decodeStructuredProjectionEnergy }
  from "../lib/webgpu-octree-structured-dynamics";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { fluidExecutionDeviceFeatures } from "../lib/gpu-startup";
import {
  OCTREE_OWNER_ARENA_MAGIC,
  OCTREE_OWNER_PAGE_CONTROL_WORDS,
} from "../lib/webgpu-octree-owner-pages";
import type { CompactOctreeFieldEvidence } from "./webgpu-smoke-compact-field";
import {
  GPUDataFlowAudit,
  type GPUDataFlowManifest,
} from "./webgpu-data-flow-manifest";
import {
  GPUCommandAudit,
  GPUPassTimestampAudit,
  auditCommandEncoder,
  writtenByteLength,
  type GPUCommandAuditReport,
  type GPUFineTimestampBucket,
  type GPUFineTimestampReport,
  type GPUPassTimestampReport,
} from "./webgpu-smoke-gpu-audits";
import {
  gravitationalPotentialEnergyProxy,
  initialSeedBrickBounds,
  inspectColumnBases,
  inspectTallVolumeGaps,
  readBufferBinding,
  readBufferBindingsPacked,
  readCompactOctreeVelocityField3D,
  readCubicVolumeField as readCubicVolumeFieldSnapshot,
  readFloatTexture2D,
  readFloatTexture3D,
  readFineUpperSurfaceField,
  readFluidBrickSnapshot,
  readGlobalFineGenerationDiagnostics,
  readSparseVoxelStats,
  readTallVelocityField3D,
  readTallVelocityTexture3D,
  readVelocityTexture3D,
  smokeRenderHybridPresentation,
  velocityDifferenceMagnitude,
  type FluidBrickSnapshot,
  type GlobalFineGenerationDiagnostics,
  type HybridPresentationSmokeStats,
  type SparseVoxelSmokeStats,
  type VelocityStageSummary,
} from "./webgpu-smoke-readbacks";
import { createPassEncoderIsolationScratch, isolateComputePassEncoders } from "./webgpu-pass-encoder-isolation";
import { encodeStructuredAuditRecordCopies, exactStructuredGenerationAuditFailures,
  finalPerformanceAuthorityFailures, STRUCTURED_GENERATION_AUDIT_SNAPSHOT,
  unpackStructuredBoundaryControl, unpackStructuredGenerationAuditSnapshot,
  unpackStructuredVelocityControl }
  from "./webgpu-smoke-structured-audit";
import { decodeOctreeMGPCGDiagnostics, octreeProjectedVariationalResidualRms,
  type OctreeMGPCGDiagnostics } from "./webgpu-smoke-pressure";
import { compactLiquidVelocityDiagnostic, compactMechanicalEnergyDiagnostic } from "./webgpu-smoke-power-diagnostics";
import { queueCompleteSimulationWall_ms } from "./webgpu-smoke-timing";
import type { PaperPhaseId, PerformanceTrace } from "../lib/performance-trace";
import type { OctreeWorkSnapshot } from "../lib/webgpu-octree-work-accounting";
import { usePerformanceInstrumentationStore } from "../lib/stores/performance-instrumentation-store";
import {
  compareScalarFields,
  compareSingleTallCellNeighborhood,
  createSmokeScenario,
  isSmokeScenarioId,
  smokeScenarioIds,
  summarizeScalarField,
  summarizeTallCellActivity,
  type ScalarFieldSummary,
  type SmokeScenario,
  type TallCellActivitySummary,
  type SmokeScenarioId
} from "./webgpu-smoke-scenarios";
import { evaluateSceneDiagnosticLane } from "../lib/scene-diagnostic-runtime";
import { sceneDiagnosticRuntimeRegistry } from "../lib/scene-diagnostic-implementations";
import type { WebGPUSmokeMethodId } from "../lib/scene-webgpu-smoke";
import { normalizeWebGPUSmokeEvidence } from "./webgpu-smoke-evidence";
import { webGPUSmokeExecutionFailures } from "./webgpu-smoke-execution-contract";
import {
  collectSceneEvidence,
} from "./scene-evidence-collector-runtime";
import { sceneEvidenceCollectorRegistry } from "./scene-evidence-collector-implementations";

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

function methodsForScenario(scenario: SmokeScenario): SimulationMethod[] {
  if (methodFilter) return methods;
  const authored = new Set<string>(scenario.lane.methods.map(({ id }) => id));
  const selected = availableMethods.filter(({ id }) => authored.has(id));
  if (selected.length !== authored.size) {
    throw new Error(`${scenario.id}/${scenario.lane.id} declares unavailable WebGPU methods: ${[...authored].join(", ")}`);
  }
  return selected;
}

interface ResolvedSceneRunOptions {
  maxDt_s?: number;
  exactSteps?: number;
  includeFinalFieldStats: boolean;
  performanceProfile: boolean;
  gpuCommandAudit: boolean;
  requireSpatialField: boolean;
  runCPUOracle: boolean;
  checkpointEvery_s: number;
  stabilityEnvelope: boolean;
  energyEverySteps: number;
  sparseStats: boolean;
  rasterCheckpoints: boolean;
  globalFineGeneration: boolean;
  powerGenerationAudit: boolean;
  powerGenerationAuditLog: boolean;
  powerAuditEverySteps: number;
  evidenceCollectors: SmokeScenario["lane"]["collect"]["evidenceCollectors"];
}

function environmentBoolean(name: string, authored: boolean): boolean {
  const value = process.env[name];
  return value === undefined ? authored : value === "1";
}

function resolveSceneRunOptions(scenario: SmokeScenario): ResolvedSceneRunOptions {
  const collect = scenario.lane.collect;
  const powerAudit = collect.powerGenerationAudit;
  return {
    maxDt_s: maxDtOverride ?? scenario.lane.stop.maxDt_s,
    exactSteps: exactStepCount ?? scenario.lane.stop.exactSteps,
    includeFinalFieldStats: process.env.FLUID_FIELD_STATS === undefined
      ? collect.fieldStats !== "none" : includeFinalFieldStats,
    performanceProfile: environmentBoolean("FLUID_PERFORMANCE_PROFILE", collect.performanceProfile),
    gpuCommandAudit: environmentBoolean("FLUID_GPU_COMMAND_AUDIT", collect.gpuCommandAudit),
    requireSpatialField: environmentBoolean("FLUID_REQUIRE_SPATIAL_FIELD", collect.spatialField),
    runCPUOracle: process.env.FLUID_CPU_ORACLE === undefined ? scenario.lane.oracle.enabled : runCPUOracle,
    checkpointEvery_s: process.env.FLUID_CHECKPOINT_EVERY_S === undefined
      ? collect.checkpointEvery_s ?? 0 : checkpointEvery_s,
    stabilityEnvelope: environmentBoolean("FLUID_STABILITY_ENVELOPE", collect.stabilityEnvelope),
    energyEverySteps: process.env.FLUID_ENERGY_EVERY_STEPS === undefined
      ? collect.energyEverySteps ?? 0 : energyEverySteps,
    sparseStats: environmentBoolean("FLUID_SPARSE_STATS", collect.sparsePublication),
    rasterCheckpoints: environmentBoolean("FLUID_RASTER_CHECKPOINTS", collect.raster === "checkpoints"),
    globalFineGeneration: environmentBoolean("FLUID_GLOBAL_FINE_GENERATION_TRANSITION", collect.globalFineGeneration),
    powerGenerationAudit: environmentBoolean("FLUID_POWER_GENERATION_AUDIT", powerAudit !== false),
    powerGenerationAuditLog: process.env.FLUID_POWER_GENERATION_AUDIT_LOG === undefined
      ? powerAudit !== false && powerAudit.log : powerGenerationAuditLog,
    powerAuditEverySteps: process.env.FLUID_POWER_AUDIT_EVERY_STEPS === undefined
      ? powerAudit === false ? 1 : powerAudit.everySteps : powerAuditEverySteps,
    evidenceCollectors: collect.evidenceCollectors,
  };
}

const qualityValue = process.env.FLUID_QUALITY ?? "balanced";
if (!["balanced", "high", "ultra"].includes(qualityValue)) throw new Error(`Unknown FLUID_QUALITY=${qualityValue}`);
const quality = qualityValue as GPUQuality;
const laneSelection = process.env.FLUID_LANE?.trim() || undefined;
const targetOverride = process.env.FLUID_TARGET_S === undefined ? undefined : Number(process.env.FLUID_TARGET_S);
const maxDtOverride = process.env.FLUID_MAX_DT === undefined ? undefined : Number(process.env.FLUID_MAX_DT);
const exactStepCount = process.env.FLUID_EXPECT_EXACT_STEPS === undefined ? undefined : Number(process.env.FLUID_EXPECT_EXACT_STEPS);
if (maxDtOverride !== undefined && (!Number.isFinite(maxDtOverride) || maxDtOverride <= 0)) throw new Error("FLUID_MAX_DT must be positive and finite");
if (exactStepCount !== undefined && (!Number.isInteger(exactStepCount) || exactStepCount < 1)) throw new Error("FLUID_EXPECT_EXACT_STEPS must be a positive integer");
if (exactStepCount !== undefined && maxDtOverride === undefined) throw new Error("FLUID_EXPECT_EXACT_STEPS requires FLUID_MAX_DT so submitted/completed time is unambiguous");
const reportEvery = Number(process.env.FLUID_REPORT_EVERY ?? 0);
const includeFinalFieldStats = process.env.FLUID_FIELD_STATS !== "0";
/** Timing-only mode keeps solver/control/timestamp readbacks while omitting
 * compact cubic reconstruction and scene quality gates. The reconstruction
 * is not part of simulationWall_ms and can independently reject a measurable
 * run when an upstream publication generation is stale. */
/** Profiling-only handshake used by the literal-first-frame xctrace lane.
 * Register the signal before construction so an early release can never take
 * Node's default SIGUSR1 action. The await itself sits after all t=0 audits and
 * immediately before recurring command accounting begins. */
const firstAdvanceProfileGate = process.env.FLUID_PROFILE_FIRST_ADVANCE_GATE === "1";
const firstAdvanceProfileGateReleased = firstAdvanceProfileGate
  ? new Promise<void>((resolve) => {
    process.once("SIGUSR1", resolve);
  })
  : undefined;
const regressionArtifactRequested = process.env.FLUID_REGRESSION_ARTIFACT === "1";
const genericPhaseTraceRequested = process.env.FLUID_GPU_FINE_TIMESTAMPS === "1";
/** X-6 needs the buffer dependency DAG without also enabling the semantic
 * phase timestamp recorder. Keep an explicit diagnostic switch so a clean
 * pass-timestamp capture cannot silently omit its manifest. */
const gpuDataFlowManifestRequested = genericPhaseTraceRequested
  || process.env.FLUID_GPU_DATA_FLOW_MANIFEST === "1";
/** One-shot, pass-local GPU timestamps for algorithm attribution. Unlike the
 * semantic fallback tracer this does not split submissions or wait between
 * phases: timestamp writes are attached to the compute passes the solver
 * already opens, and the first recurring command buffer is resolved only
 * after it has been submitted. */
const gpuPassTimestampRequested = process.env.FLUID_GPU_PASS_TIMESTAMPS === "1";
/** Terminal-only SPGrid hierarchy census. Unlike pass timestamps this does not
 * instrument, split, or relabel any recurring GPU work. */
const spgridHierarchyCensusRequested =
  process.env.FLUID_SPGRID_HIERARCHY_CENSUS === "1";
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
const dataFlowSkipAdvances = Math.max(0,
  Math.floor(Number(process.env.FLUID_GPU_DATA_FLOW_SKIP_ADVANCES ?? 0)));
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
const octreeSurfaceGradingOverride = process.env.FLUID_OCTREE_SURFACE_GRADING === undefined
  ? undefined : Number(process.env.FLUID_OCTREE_SURFACE_GRADING);
if (octreeSurfaceGradingOverride !== undefined
  && (!Number.isInteger(octreeSurfaceGradingOverride)
    || octreeSurfaceGradingOverride < 1 || octreeSurfaceGradingOverride > 4)) {
  throw new Error("FLUID_OCTREE_SURFACE_GRADING must be an integer from 1 through 4");
}
// Section 5 surface-band thickness, independent of the pressure band above.
// Unset leaves the projection on the pressure band, so an A/B against a lane's
// recorded numbers only needs this one variable.
const octreeFineBandOverride = process.env.FLUID_OCTREE_FINE_BAND === undefined
  ? undefined : Number(process.env.FLUID_OCTREE_FINE_BAND);
// A band narrower than the pressure band leaves the fine-to-coarse restriction
// without valid phi where the pressure cell centres read it, and the generation
// is rejected. That is a mismatch with the pressure band rather than a floor on
// this value, so the harness admits 1 and lets the pairing fail closed.
if (octreeFineBandOverride !== undefined
  && (!Number.isInteger(octreeFineBandOverride) || octreeFineBandOverride < 1)) {
  throw new Error("FLUID_OCTREE_FINE_BAND must be a positive integer");
}
const octreePressureRowCapacityOverride = process.env.FLUID_PRESSURE_ROW_CAPACITY === undefined
  ? undefined : Number(process.env.FLUID_PRESSURE_ROW_CAPACITY);
if (octreePressureRowCapacityOverride !== undefined
  && (!Number.isSafeInteger(octreePressureRowCapacityOverride) || octreePressureRowCapacityOverride < 1)) {
  throw new Error("FLUID_PRESSURE_ROW_CAPACITY must be a positive integer");
}
/** Lower Dawn's negotiated storage-binding ceiling to reproduce a browser
 * adapter tier exactly. The request stays clamped to the native adapter and
 * affects the device limit seen by the production allocation path. */
const storageBindingLimitOverride = process.env.FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES === undefined
  ? undefined : Number(process.env.FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES);
if (storageBindingLimitOverride !== undefined
  && (!Number.isSafeInteger(storageBindingLimitOverride) || storageBindingLimitOverride < 1)) {
  throw new Error("FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES must be a positive integer");
}
// Stage-1 coarse-baseline spelling. Keep the established long name as a
// compatibility alias for existing lanes; the explicit short name wins so a
// caller can override a lane table that still pins the paper's factor 4.
const octreeGlobalFineFactorOverride =
  process.env.FLUID_FINE_FACTOR ?? process.env.FLUID_OCTREE_GLOBAL_FINE_FACTOR;
if (octreeGlobalFineFactorOverride !== undefined && !["1", "4", "8"].includes(octreeGlobalFineFactorOverride)) {
  throw new Error("FLUID_FINE_FACTOR/FLUID_OCTREE_GLOBAL_FINE_FACTOR must be 1, 4, or 8");
}
const powerGenerationAuditLog = process.env.FLUID_POWER_GENERATION_AUDIT_LOG !== "0";
const powerCandidateAuditRequested = process.env.FLUID_POWER_CANDIDATE_AUDIT === "1";
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
const energyEverySteps = Number(process.env.FLUID_ENERGY_EVERY_STEPS ?? 0);
const globalFineGenerationTransitionRequested = process.env.FLUID_GLOBAL_FINE_GENERATION_TRANSITION === "1";
const octreeTopologyCensusRequested = process.env.FLUID_OCTREE_TOPOLOGY_CENSUS === "1";
// Publication-transition acceptance needs the existing bounded renderer
// counter readback so it can distinguish global fine/coarse authority from an
// adaptive or retained presentation fallback. This is QA-only and adds no
// shader bindings or simulation readback.
if (globalFineGenerationTransitionRequested) process.env.FLUID_WATER_DIAGNOSTICS = "1";
if (!Number.isInteger(energyEverySteps) || energyEverySteps < 0) throw new Error("FLUID_ENERGY_EVERY_STEPS must be a non-negative integer");
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
  const selection = process.env.FLUID_SCENE ?? "all";
  if (selection === "all") return [...smokeScenarioIds];
  const ids = selection.split(",").map((value) => value.trim()).filter(Boolean);
  for (const id of ids) if (!isSmokeScenarioId(id)) throw new Error(`Unknown FLUID_SCENE=${id}; expected all or ${smokeScenarioIds.join(", ")}`);
  return ids as SmokeScenarioId[];
}

function applySceneOverrides(scene: SceneDescription, resolvedMaxDt_s = maxDtOverride): SceneDescription {
  if (resolvedMaxDt_s !== undefined) scene.numerics.maxDt_s = resolvedMaxDt_s;
  return scene;
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
  finalPerformanceAuthority?: Readonly<Record<string, unknown>>;
  algorithmDiagnostics?: Record<string, unknown>;
  gpuDataFlowManifest?: GPUDataFlowManifest;
  /** Accepted steps whose live power topology, faces, transfer, and MGPCG publication passed the generation audit. */
  powerGenerationAuditedSteps: number;
  /** Iteration envelope decoded from the terminal packed generation audit;
   * this adds no readback beyond that single aggregate snapshot. */
  mgpcgIterationAudit?: { samples: number; minimum: number; maximum: number; histogram: Record<string, number> };
  velocitySummary?: VelocityStageSummary;
  /** Values produced by terminal scene-declared collectors. */
  terminalEvidence: Readonly<Record<string, unknown>>;
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
  /** Capability labels successfully published by scene-declared collectors. */
  collectedEvidence: string[];
  checkpoints: Array<{
    time_s: number;
    field: Float32Array;
    summary: ScalarFieldSummary;
    /** GPU raster result sampled for QA only; never feeds the simulation. */
    raster?: HybridPresentationSmokeStats;
    globalFineGeneration?: GlobalFineGenerationDiagnostics;
    preProjectionVelocity?: Float32Array;
    postProjectionVelocity?: Float32Array;
    /** Values produced by scene-declared collectors, keyed by collector ID. */
    evidence?: Readonly<Record<string, unknown>>;
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

function reportResult(scenario: SmokeScenario, result: GPUSmokeResult) {
  const scenarioId = scenario.id;
  const info = result.info;
  console.log(JSON.stringify({
    scenario: scenarioId, method: result.method, phase: "result", construction_ms: Math.round(result.construction_ms), runtime_ms: Math.round(result.runtime_ms), simulationWall_ms: Math.round(result.simulationWall_ms), steps: result.steps,
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
    velocitySummary: result.velocitySummary,
    initialFluidBrickStats: result.initialFluidBrickStats,
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
        ceilingContactPixels: raster.ceilingContactPixels,
        wallCornerCapPixels: raster.wallCornerCapPixels,
        wallCornerMaximumY_m: raster.wallCornerMaximumY_m,
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
  scenario: SmokeScenario,
  method: SimulationMethod,
  target_s: number,
  oracleSteps: number,
  options: ResolvedSceneRunOptions,
): Promise<GPUSmokeResult> {
  const scenarioId = scenario.id, scene = applySceneOverrides(scenario.scene, options.maxDt_s);
  const exactStepCount = options.exactSteps;
  const maxDtOverride = options.maxDt_s;
  const includeFinalFieldStats = options.includeFinalFieldStats;
  const performanceProfileRequested = options.performanceProfile;
  const gpuCommandAuditRequested = options.gpuCommandAudit;
  const readCubicVolumeField = (readDevice: GPUDevice, solver: GPUSolverInstance) =>
    readCubicVolumeFieldSnapshot(readDevice, solver, options.requireSpatialField);
  const checkpointEvery_s = options.checkpointEvery_s;
  const stabilityEnvelopeRequested = options.stabilityEnvelope;
  const energyEverySteps = options.energyEverySteps;
  const sparseStatsRequested = options.sparseStats;
  const rasterCheckpointRequested = options.rasterCheckpoints;
  const globalFineGenerationTransitionRequested = options.globalFineGeneration;
  const powerGenerationAuditRequested = options.powerGenerationAudit;
  const powerGenerationAuditLog = options.powerGenerationAuditLog;
  const powerAuditEverySteps = options.powerAuditEverySteps;
  const evidenceCollectors = options.evidenceCollectors;
  const declaredDtPattern = scenario.lane.stop.dtPattern_s;
  const regressionDtPattern = Array.isArray(declaredDtPattern)
    ? declaredDtPattern.filter((value): value is number => typeof value === "number"
      && Number.isFinite(value) && value > 0)
    : [];
  const perturbCadence = regressionDtPattern.length > 0;
  const collectStabilityEnvelope = perturbCadence || stabilityEnvelopeRequested;
  const applicableTerminalCollectors = evidenceCollectors.filter((collector) => collector.phase === "terminal"
    && (!collector.methods || collector.methods.includes(method.id as WebGPUSmokeMethodId)));
  const terminalSources = new Set(applicableTerminalCollectors.flatMap((collector) => collector.requires ?? []));
  const applicableCheckpointCollectors = evidenceCollectors.filter((collector) => collector.phase === "checkpoint"
    && (!collector.methods || collector.methods.includes(method.id as WebGPUSmokeMethodId)));
  const checkpointSources = new Set(applicableCheckpointCollectors.flatMap((collector) => collector.requires ?? []));
  const authoredProfile = scenario.lane.methods.find((entry) => entry.id === method.id);
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
  if (storageBindingLimitOverride !== undefined) {
    requiredLimits.maxStorageBufferBindingSize = Math.min(
      requiredLimits.maxStorageBufferBindingSize,
      storageBindingLimitOverride,
    );
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  await assertComputeSentinel(device);
  let lost: GPUDeviceLostInfo | undefined;
  void device.lost.then((info) => { lost = info; });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  const commandAudit = gpuCommandAuditRequested ? new GPUCommandAudit() : undefined;
  const dataFlowAudit = gpuDataFlowManifestRequested ? new GPUDataFlowAudit() : undefined;
  const requestedPassTimestampQueryCapacity = Number(
    process.env.FLUID_GPU_PASS_TIMESTAMP_QUERY_CAPACITY ?? 2048);
  const passTimestampQueryCapacity = Number.isFinite(requestedPassTimestampQueryCapacity)
    ? Math.max(2, 2 * Math.floor(requestedPassTimestampQueryCapacity / 2)) : 2048;
  const passTimestampLabelPrefixes = (process.env.FLUID_GPU_PASS_TIMESTAMP_LABEL_PREFIXES ?? "")
    .split(",").map((value) => value.trim()).filter((value) => value.length > 0);
  const passTimestampAudit = gpuPassTimestampRequested
    && device.features.has("timestamp-query")
    ? new GPUPassTimestampAudit(device, Math.max(1,
      Math.floor(Number(process.env.FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS ?? 1))),
    passTimestampQueryCapacity, gpuIsolatePassEncodersRequested, gpuIsolatePassLabelsRequested,
    Math.max(0, Math.floor(Number(process.env.FLUID_GPU_PASS_TIMESTAMP_SKIP_COMMAND_BUFFERS ?? 0))),
    passTimestampLabelPrefixes)
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
        const shaderModule = target.createShaderModule(descriptor);
        dataFlowAudit?.registry.recordShader(shaderModule, descriptor);
        return shaderModule;
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
  if (method.id === "octree" && maximumLeafSizeOverride !== undefined) values.maximumLeafSize = maximumLeafSizeOverride;
  if (method.id === "octree" && octreeInterfaceBandOverride !== undefined) {
    values.interfaceRefinementBandCells = octreeInterfaceBandOverride;
  }
  if (method.id === "octree" && octreeSurfaceGradingOverride !== undefined) {
    values.surfaceRefinementGradingLayers = octreeSurfaceGradingOverride;
  }
  if (method.id === "octree" && octreeFineBandOverride !== undefined) {
    values.fineLevelSetBandCells = octreeFineBandOverride;
  }
  if (method.id === "octree" && octreeGlobalFineFactorOverride !== undefined) values.globalFineLevelSetFactor = octreeGlobalFineFactorOverride;
  // Factor one transports phi directly on the accepted octree rows and owns
  // no separate sparse fine-band publication.
  const hasSeparateFineLevelSetBand = method.id === "octree"
    && Number(values.globalFineLevelSetFactor) !== 1;
  const verifyGlobalFineGenerationTransition = globalFineGenerationTransitionRequested
    && hasSeparateFineLevelSetBand;
  if (method.id === "octree" && octreePressureRowCapacityOverride !== undefined) {
    values.pressureRowCapacity = octreePressureRowCapacityOverride;
  }
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
  // The structured dynamics instance decides whether to encode its projection-
  // energy summaries while it is constructed. Authored lane collection is
  // resolved above without rewriting process.env, so propagate that resolved
  // requirement across construction or every later audit snapshot contains the
  // intentionally unpublished zero buffer. Preserve an explicit caller override.
  const previousStructuredEnergyProbe = process.env.FLUID_STRUCTURED_ENERGY_PROBE;
  const enableAuthoredStructuredEnergyProbe = previousStructuredEnergyProbe === undefined
    && method.id === "octree"
    && (powerGenerationAuditRequested || collectStabilityEnvelope || energyEverySteps > 0);
  if (enableAuthoredStructuredEnergyProbe) process.env.FLUID_STRUCTURED_ENERGY_PROBE = "1";
  let solver: GPUSolverInstance;
  try {
    solver = probeLayout
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
  } finally {
    if (enableAuthoredStructuredEnergyProbe) delete process.env.FLUID_STRUCTURED_ENERGY_PROBE;
  }
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
  const initialGlobalFineGeneration = verifyGlobalFineGenerationTransition && method.id === "octree"
    ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
  const initialGlobalFineRaster = verifyGlobalFineGenerationTransition && method.id === "octree"
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
  if (firstAdvanceProfileGateReleased) {
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "before-first-advance", profileGate: "waiting-for-sigusr1" }));
    await firstAdvanceProfileGateReleased;
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "before-first-advance", profileGate: "released" }));
  }
  // Construction and t=0 publication have separate costs. The command audit
  // below measures only recurring advance work and explicitly requested
  // profiler/readback activity after the initialized solver is warm.
  commandAudit?.reset();
  if (dataFlowSkipAdvances === 0) dataFlowAudit?.start();
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
  const checkpoints: GPUSmokeResult["checkpoints"] = [];
  const collectedEvidence = new Set<string>();
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
    coarseOffsetBytes: 144, coarseBytes: 64,
    fineHeaderOffsetBytes: 208, fineHeaderBytes: 28,
    strideBytes: 240,
  });
  /** The benchmark and acceptance lanes must evaluate every tripwire. Any
   * other octree run captures them opportunistically: a trip still fails, but
   * a scene with no compact fine authority is "not applicable" rather than a
   * wiring failure. */
  const tripwiresRequired = !tripwiresDisabled && method.id === "octree"
    && (tripwiresForcedRequired || powerGenerationAuditRequested || performanceProfileRequested);
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
      globalFineSummaryDebug?: { coarseControl: GPUBuffer };
      workAccountingBuffers?: { fineTopologyControl?: GPUBufferBinding };
    };
    return {
      topology: authority.workAccountingBuffers?.fineTopologyControl,
      restriction: authority.globalFineRestrictionControl,
      mgpcg: authority.mgpcgControl,
      coarse: authority.globalFineSummaryDebug?.coarseControl,
      fineWorklist: authority.globalFineLevelSetSource?.worklist,
    };
  };
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
    if (dataFlowAudit) {
      if (steps >= dataFlowSkipAdvances
        && steps < dataFlowSkipAdvances + genericPhaseTraceAdvances) dataFlowAudit.start();
      else dataFlowAudit.stop();
    }
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
        encoder.copyBufferToBuffer(sources.coarse!, 0, tripwireSnapshot,
          base + TRIPWIRE_RECORD.coarseOffsetBytes, TRIPWIRE_RECORD.coarseBytes);
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
      await awaitAdvanceCompletion();
      // The drain is solver execution, including on lanes shorter than the
      // periodic fence cadence. Only the readbacks after it are diagnostics.
      const samplingStartedAt = performance.now();
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
      await awaitAdvanceCompletion();
      const samplingStartedAt = performance.now();
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
      await awaitAdvanceCompletion();
      const samplingStartedAt = performance.now();
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
      let compactVelocityField: Float32Array | undefined;
      if (method.id === "octree" && initialPotentialEnergyProxy !== undefined) {
        const compact = await readCompactOctreeVelocityField3D(device, solver,
          [solver.info.nx, solver.info.ny, solver.info.nz]);
        if (compact) {
          compactVelocityField = compact.field;
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
          verifyGlobalFineGenerationTransition)
        : undefined;
      const globalFineGeneration = verifyGlobalFineGenerationTransition && method.id === "octree"
        ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
      const fineUpperSurfaceField = checkpointSources.has("fine upper surface") && method.id === "octree"
        ? await readFineUpperSurfaceField(device, solver,
          [solver.info.nx, solver.info.ny, solver.info.nz])
        : undefined;
      const collected = collectSceneEvidence(sceneEvidenceCollectorRegistry, evidenceCollectors, "checkpoint", {
        scene, method: method.id as WebGPUSmokeMethodId, grid: [solver.info.nx, solver.info.ny, solver.info.nz],
        time_s: solver.info.submittedTime_s ?? 0, volumeField: cubic.field,
        ...(compactVelocityField ? { velocityField: compactVelocityField } : {}),
        ...(fineUpperSurfaceField ? { fineUpperSurfaceField } : {}),
      });
      for (const capability of collected.available) collectedEvidence.add(capability);
      checkpoints.push({ time_s: solver.info.submittedTime_s ?? 0, field: cubic.field, summary: cubic.summary,
        raster, globalFineGeneration, preProjectionVelocity, postProjectionVelocity, compactMechanicalEnergy,
        ...(Object.keys(collected.values).length > 0 ? { evidence: collected.values } : {}) });
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
  dataFlowAudit?.stop();
  await awaitAdvanceCompletion();
  const simulationCompletedAt_ms = performance.now();
  const simulationWall_ms = queueCompleteSimulationWall_ms(
    runStarted, simulationCompletedAt_ms, samplingWall_ms,
  );
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
        // Aanjaneya et al. 2017 Section 5
        // (`docs/papers/aanjaneya-2017-power-liquids.txt`) requires the separate
        // fine-SPGrid publication, not this project's optional enclosed-volume
        // correction (the producer itself documents that extension as
        // project-specific). Enforce its receipt only when the caller requests
        // the volume/stability envelope; it is not pressure/fine authority.
        if (stabilityEnvelope && !volumeValid) {
          generationFailures.push("fine volume publication is invalid or stale");
        }
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
        topologyControl?: readonly number[];
        worklistHeader?: readonly number[];
        coarseControl?: readonly number[];
        fineRestrictionControl?: readonly number[];
        airSupportControl?: readonly number[];
        precedingAirSupportTerminal?: readonly number[];
        firstAirSupportFailure?: readonly number[];
        airSupportFailureTopology?: Readonly<Record<string, unknown>>;
        structuredRejectCarry?: readonly number[];
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
          topology: fineFailure.topologyControl,
          fineWorklist: fineFailure.worklistHeader,
          coarse: fineFailure.coarseControl,
          restriction: fineFailure.fineRestrictionControl,
          control: fineFailure.airSupportControl,
          precedingTerminal: fineFailure.precedingAirSupportTerminal,
          firstFailure: fineFailure.firstAirSupportFailure,
          topologyFailure: fineFailure.airSupportFailureTopology,
          structuredRejectCarry: fineFailure.structuredRejectCarry,
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
          // Aanjaneya et al. 2017 Section 5
          // (`docs/papers/aanjaneya-2017-power-liquids.txt`) gives the fine
          // SPGrid and background octree independent lifecycles. A known,
          // explicit rollback is the provenance for retaining the latter; it
          // is not a silent failure. Unknown or incomplete rollback controls
          // still trip immediately.
          const retainedBackgroundOctree = fineTopologyRetainsBackgroundOctree(topologyControl);
          if (topology.rolledBack && !retainedBackgroundOctree) {
            trip("topology-rollback", { rolledBack: true,
              flags: topology.flags, published: topology.published,
              downstreamFinalizeReason: topology.downstreamFinalizeReason,
              interfaceBricks: topology.interfaceBricks,
              desiredBricks: topology.desiredBricks,
              activatedBricks: topology.activatedBricks,
              control: Array.from(topologyControl) });
          }
          // 2. Section 5 restriction authority. Aanjaneya et al. deliberately
          //    keep the fine SPGrid only around the surface; the background
          //    octree owns every other row. A global uncovered-row fraction is
          //    therefore scene-dependent and invalid (the ocean legitimately
          //    leaves about 91% of coarse rows outside the fine band). Audit
          //    the real two-authority receipt instead: every accepted fine row
          //    becomes one coarse correction and those rows cover the complete
          //    interface set.
          const restrictionWords = words(record, TRIPWIRE_RECORD.restrictionOffsetBytes,
            TRIPWIRE_RECORD.restrictionBytes);
          const restriction = unpackFineToCoarseGPUControl(restrictionWords);
          const coarseWords = words(record, TRIPWIRE_RECORD.coarseOffsetBytes,
            TRIPWIRE_RECORD.coarseBytes);
          const coarse = unpackOctreePowerCoarseLevelSetControl(coarseWords);
          const restrictionAudit = auditSection5FineRestriction(restriction, coarse);
          if (restrictionAudit.failure && hasSeparateFineLevelSetBand
            && !retainedBackgroundOctree) {
            trip("restriction-unaccepted", { reason: restrictionAudit.failure,
              ...restrictionAudit, restrictionControl: Array.from(restrictionWords),
              coarseControl: Array.from(coarseWords) });
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
              rows: mgpcg.rows, firstErrorStage: mgpcg.firstErrorStage,
              firstErrorRow: mgpcg.firstErrorRow, control: Array.from(mgpcgWords) });
          } else if (mgpcg.iterations > 0 && !mgpcg.converged) {
            trip("mgpcg-nonconvergence", { converged: false, iterations: mgpcg.iterations,
              rows: mgpcg.rows, relativeResidual: mgpcg.relativeResidual,
              residualSquared: mgpcg.residualSquared, rhsSquared: mgpcg.rhsSquared });
          }
          // 4. Fine-band capacity overflow. Older publishers degraded the
          //    active count to INVALID. The transactional publisher instead
          //    retains the prior fine authority and reports the rejected
          //    required count in topology control, so both receipts must be
          //    checked. The count is worklist header word ONE; a prior consumer
          //    read word zero (the generation) and printed nonsense.
          const header = readFineLevelSetWorksetHeader(words(record,
            TRIPWIRE_RECORD.fineHeaderOffsetBytes, TRIPWIRE_RECORD.fineHeaderBytes));
          if (header === undefined) {
            trip("fine-band-sentinel", { unevaluable: true,
              reason: "fine worklist header could not be decoded" });
          } else if (header.activeCount === 0xffff_ffff) {
            trip("fine-band-sentinel", { activeCount: header.activeCount,
              sentinel: "0xFFFFFFFF", capacity: header.capacity,
              generation: header.generation, flags: header.flags });
          } else if ((topology.flags & FINE_LEVELSET_TOPOLOGY_ERROR.capacity) !== 0
            || topology.requiredDesiredBricks > header.capacity) {
            trip("fine-band-sentinel", { activeCount: header.activeCount,
              retainedPublication: topology.rolledBack,
              requiredDesiredBricks: topology.requiredDesiredBricks,
              requiredDesiredBricksExact: topology.requiredDesiredBricksExact,
              capacity: header.capacity, generation: header.generation,
              topologyFlags: topology.flags });
          }
        }
      } finally {
        if (tripwireSnapshot.mapState === "mapped") tripwireSnapshot.unmap();
        tripwireSnapshot.destroy();
      }
    }
    const allowed = tripped.filter((entry) => tripwireAllowList.has(entry.id));
    const failing = tripped.filter((entry) => !tripwireAllowList.has(entry.id));
    if (process.env.FLUID_TRIPWIRE_ALLOW_SUMMARY === "1" && allowed.length > 0) {
      const counts: Record<string, number> = {};
      for (const entry of allowed) counts[entry.id] = (counts[entry.id] ?? 0) + 1;
      console.error(`[tripwires ALLOWED probe summary] ${JSON.stringify({ counts,
        firstTrips: allowed.slice(0, 12), lastTrip: allowed[allowed.length - 1] })}`);
    } else {
      for (const entry of allowed) {
        console.error(`[tripwire ${entry.id} ALLOWED] ${JSON.stringify(entry)}`);
      }
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
  // Finalized after the pass-local timestamp audit drains. Semantic phase
  // labels ("fine-sdf-redistance: ...") do not match compute-pass labels and
  // would silently assign zero duration to every DAG node.
  let gpuDataFlowManifest: ReturnType<GPUDataFlowAudit["report"]> | undefined;
  let finalPerformanceAuthority: Readonly<Record<string, unknown>> | undefined;
  if (performanceProfileRequested && method.id === "octree") {
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
    finalPerformanceAuthority = finalAuthority;
    console.log(JSON.stringify({
      scenario: scenarioId, method: resultMethod,
      phase: "final-performance-authority", ...finalAuthority,
    }));
    if (finalAuthorityFailures.length !== 0) {
      if (process.env.FLUID_QUALITY_INVALID_PROBE === "1") {
        console.error(`[final performance authority ALLOWED probe] ${JSON.stringify(finalAuthority)}`);
      } else {
        throw new Error(`final performance authority rejected: ${JSON.stringify(finalAuthority)}`);
      }
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
      octreeProjection?: {
        readSolveDiagnostics(): Promise<void>;
        readTopologyLeafCensus(): Promise<{
          generation: number;
          residentOwnerPages: number;
          topologyLeaves: number;
          representedCells: number;
          leafCountsBySize: Readonly<Record<string, number>>;
        }>;
      };
    }).octreeProjection;
    if (projection) {
      await projection.readSolveDiagnostics();
      if (octreeTopologyCensusRequested) {
        const census = await projection.readTopologyLeafCensus();
        console.log(JSON.stringify({
          scenario: scenarioId,
          method: method.id,
          phase: "octree-topology-census",
          ...census,
        }));
      }
    }
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
  let terminalCompactVelocity: Awaited<ReturnType<typeof readCompactOctreeVelocityField3D>>;
  if (method.id === "octree" && info.powerDiagramAuthoritative === true && final
    && (!Number.isFinite(info.maxSpeed_m_s ?? NaN)
      || !Number.isFinite(info.maxComponentCfl ?? NaN))) {
    terminalCompactVelocity = await readCompactOctreeVelocityField3D(
      device, solver, [info.nx, info.ny, info.nz],
    );
    if (terminalCompactVelocity) {
      const spacing = [scene.container.width_m / info.nx,
        scene.container.height_m / info.ny,
        scene.container.depth_m / info.nz] as const;
      const measuredDt = (info.simulatedTime_s ?? 0) / Math.max(1, steps);
      const terminalDt = powerGenerationAuditDts.at(-1)
        ?? (measuredDt > 0 ? measuredDt : scene.numerics.maxDt_s);
      const velocity = compactLiquidVelocityDiagnostic(
        terminalCompactVelocity.field, final.field,
        spacing[0] * spacing[1] * spacing[2], spacing, terminalDt,
      );
      info.maxSpeed_m_s = velocity.maximumLiquidComponentSpeed_m_s;
      info.maxComponentCfl = velocity.maximumLiquidComponentCfl;
      info.nonFiniteCount = (info.nonFiniteCount ?? 0) + velocity.nonFiniteLiquidComponentCount;
    }
  }
  if (terminalSources.has("compact velocity") && method.id === "octree"
    && info.powerDiagramAuthoritative === true && final && !terminalCompactVelocity) {
    terminalCompactVelocity = await readCompactOctreeVelocityField3D(
      device, solver, [info.nx, info.ny, info.nz],
    );
  }
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
  let terminalVelocityField: Float32Array | undefined;
  let terminalCompactVelocityEvidence: Readonly<Record<string, unknown>> | undefined;
  if (terminalSources.has("collocated velocity") && final) {
    if (method.id === "tall-cell" && velocityTexture && info.gridKind === "restricted-tall-cell") {
      const bases = await readFloatTexture2D(device, solver.columnBaseTexture!, info.nx, info.nz);
      terminalVelocityField = await readTallVelocityField3D(
        device, velocityTexture, info.nx, info.storedNy, info.nz, info.ny, bases,
      );
    } else if (method.id === "octree") {
      const compact = terminalCompactVelocity
        ?? await readCompactOctreeVelocityField3D(device, solver, [info.nx, info.ny, info.nz]);
      if (compact) {
        const { field, ...evidence } = compact;
        terminalVelocityField = field;
        terminalCompactVelocityEvidence = evidence;
      }
    }
  }
  terminalVelocityField ??= terminalSources.has("compact velocity") ? terminalCompactVelocity?.field : undefined;
  const terminalCollected = final
    ? collectSceneEvidence(sceneEvidenceCollectorRegistry, evidenceCollectors, "terminal", {
      scene, method: method.id as WebGPUSmokeMethodId, grid: [info.nx, info.ny, info.nz],
      time_s: info.simulatedTime_s ?? 0, volumeField: final.field,
      ...(terminalVelocityField ? { velocityField: terminalVelocityField } : {}),
      ...(terminalCompactVelocityEvidence ? { compactVelocityEvidence: terminalCompactVelocityEvidence } : {}),
    })
    : { values: {}, available: [] };
  for (const capability of terminalCollected.available) collectedEvidence.add(capability);
  const hybridPresentationStats = sparseStatsRequested && method.id === "octree"
    ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies)
    : undefined;
  // Always captured for octree: the structured-validation gates require the
  // final generation diagnostics, and a gate that reads `undefined` reports
  // a wiring failure rather than evaluating the solver's actual state.
  const finalGlobalFineGeneration = method.id === "octree"
    ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
  const finalGlobalFineRaster = verifyGlobalFineGenerationTransition && method.id === "octree"
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
    ? await readBufferBinding(device, { buffer: finalSolver.mgpcgControl }, 104)
    : undefined;
  const octreeMGPCGDiagnostics = mgpcgControlBytes
    ? decodeOctreeMGPCGDiagnostics(new Uint32Array(mgpcgControlBytes.buffer, mgpcgControlBytes.byteOffset, 26))
    : undefined;
  const accurateClassDispatchBinding = (finalSolver as GPUSolverInstance & {
    workAccountingBuffers?: { accurateClassDispatch?: GPUBufferBinding };
  }).workAccountingBuffers?.accurateClassDispatch;
  const accurateClassDispatchBytes = accurateClassDispatchBinding
    ? await readBufferBinding(device, accurateClassDispatchBinding, 29 * 4)
    : undefined;
  if (accurateClassDispatchBytes && octreeMGPCGDiagnostics?.flags) {
    console.error("[A2 dispatch diagnostics]", Array.from(new Uint32Array(
      accurateClassDispatchBytes.buffer, accurateClassDispatchBytes.byteOffset, 29)));
  }
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
  gpuDataFlowManifest = dataFlowAudit?.report(
    Math.min(Math.max(0, steps - dataFlowSkipAdvances), genericPhaseTraceAdvances),
    gpuPassTimestamps?.byLabel ?? gpuFineTimestamps?.byLabel,
  );
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
      readSPGridHierarchyCensus(): Promise<{
        levels: readonly Readonly<Record<string, number>>[];
      } | undefined>;
    };
    workAccountingPlan?: Record<string, unknown>;
    globalFineLevelSetSource?: WebGPUFineLevelSetBrickSource;
  };
  const terminalAlgorithmState = gpuPassTimestampRequested
    ? await diagnosticProjection.octreeProjection?.readGlobalFineLevelSetDiagnostics()
    : undefined;
  const spgridHierarchy = (gpuPassTimestampRequested || spgridHierarchyCensusRequested)
    ? await diagnosticProjection.octreeProjection?.readSPGridHierarchyCensus()
    : undefined;
  const finePlan = diagnosticProjection.globalFineLevelSetSource?.plan;
  const algorithmDiagnostics = terminalAlgorithmState || spgridHierarchy ? {
    topologyControl: terminalAlgorithmState?.topologyControl,
    structuredVelocityControl: terminalAlgorithmState?.structuredVelocityControl,
    structuredBoundaryControl: terminalAlgorithmState?.structuredBoundaryControl,
    airSupportControl: terminalAlgorithmState?.airSupportControl,
    airSupportTerminalScratch: terminalAlgorithmState?.airSupportTerminalScratch,
    finePageDeltaHeader: terminalAlgorithmState?.finePageDeltaHeader,
    finePlan: finePlan ? {
      maximumResidentBricks: finePlan.maximumResidentBricks,
      logicalBrickCount: finePlan.logicalBrickCount,
      samplesPerBrick: finePlan.samplesPerBrick,
      brickResolution: finePlan.brickResolution,
      fineFactor: finePlan.fineFactor,
    } : undefined,
    pressurePlan: diagnosticProjection.workAccountingPlan,
    spgridHierarchy,
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
    finalPerformanceAuthority,
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
    terminalEvidence: terminalCollected.values,
    initialFluidBrickStats, sparseVoxelStats, hybridPresentationStats,
    initialGlobalFineGeneration, initialGlobalFineRaster, finalGlobalFineGeneration, finalGlobalFineRaster,
    octreePowerTopologyDiagnostics,
    octreeMGPCGDiagnostics,
    stabilityEnvelope,
    collectedEvidence: [...collectedEvidence],
    octreeWorkAccounting: capturedWorkAccounting?.snapshot
      ?? accountingOwner.workAccounting?.snapshot(),
    octreeWorkAccountingBlocker: capturedWorkAccounting?.pressure?.blocker,
    energyTrace, checkpoints
  };
  reportResult(scenario, result);
  solver.destroy(); device.destroy();
  return result;
}

function cpuField(solver: EulerianFluidSolver) {
  const field = new Float32Array(solver.fluid.length);
  for (let index = 0; index < field.length; index += 1) field[index] = solver.fluid[index] ? 1 : 0;
  return field;
}

function runMatchedCPUOracle(scenario: SmokeScenario, grid: [number, number, number], oracleSteps: number,
  options: ResolvedSceneRunOptions) {
  const scenarioId = scenario.id;
  const cellCount = grid[0] * grid[1] * grid[2];
  if (cpuMaximumCells > 0 && cellCount > cpuMaximumCells) {
    console.log(JSON.stringify({ scenario: scenarioId, method: "cpu-reference", phase: "oracle-skipped", cubicGrid: grid, cellCount, reason: `exact grid exceeds FLUID_CPU_MAX_CELLS=${cpuMaximumCells}; set 0 for unlimited` }));
    return undefined;
  }
  const scene = applySceneOverrides(scenario.scene, options.maxDt_s), started = performance.now();
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

const failures: string[] = [];
try {
  for (const scenarioId of selectedScenarios()) {
    const scenario = createSmokeScenario(scenarioId, laneSelection);
    const scenarioMethods = methodsForScenario(scenario);
    const runOptions = resolveSceneRunOptions(scenario);
    applySceneOverrides(scenario.scene, runOptions.maxDt_s);
    if (runOptions.globalFineGeneration) process.env.FLUID_WATER_DIAGNOSTICS = "1";
    const oracleSteps = Math.max(1, Math.round(oracleStepsOverride ?? scenario.oracleSteps));
    const target_s = Math.max(targetOverride ?? scenario.target_s, oracleSteps * scenario.scene.numerics.maxDt_s);
    console.log(JSON.stringify({ scenario: scenarioId, lane: scenario.lane.id, phase: "scenario", description: scenario.description, target_s, oracleSteps, quality, methods: scenarioMethods.map((method) => method.id), cpuOracle: runOptions.runCPUOracle }));
    if (scenarioMethods.some((method) => method.id === "tall-cell")) {
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
    for (const method of scenarioMethods) results.push(await runGPU(scenario, method, target_s, oracleSteps, runOptions));
    const executionFailures = webGPUSmokeExecutionFailures(results, {
      exactSteps: runOptions.exactSteps,
      maxDt_s: runOptions.maxDt_s,
    });
    const diagnosticEvidence = normalizeWebGPUSmokeEvidence(results.map((result) => ({
      ...result,
      energyTraceSummary: energyTraceSummary(result.energyTrace),
    })));
    const diagnosticEvaluation = evaluateSceneDiagnosticLane(sceneDiagnosticRuntimeRegistry, {
      scene: applySceneOverrides(scenario.scene, runOptions.maxDt_s),
      lane: scenario.lane,
      evidence: diagnosticEvidence,
    });
    console.log(JSON.stringify({
      scenario: scenarioId,
      lane: scenario.lane.id,
      phase: "diagnostic-evaluation",
      passed: (runOptions.performanceProfile || diagnosticEvaluation.passed)
        && executionFailures.length === 0,
      enforced: !runOptions.performanceProfile,
      executionFailures,
      findings: diagnosticEvaluation.findings,
    }));
    // Timing-only runs are already gated in runGPU by the packed terminal
    // authority controls immediately after the measured window. The profiler
    // deliberately disables expensive scene evidence collectors, so their
    // absent findings are diagnostic output, not a physics failure. Exact-step
    // execution remains enforced here.
    const enforcedDiagnosticFindings = runOptions.performanceProfile
      ? []
      : diagnosticEvaluation.findings.filter((finding) => !finding.passed
        && finding.severity === "error");
    const scenarioDiagnosticFailures = [
      ...executionFailures.map((failure) => `${scenarioId}: execution: ${failure}`),
      ...enforcedDiagnosticFindings
        .map((finding) => `${scenarioId}: diagnostic ${finding.checkId}`
          + `${finding.method ? ` (${finding.method})` : ""}: ${finding.message}`),
    ];
    failures.push(...scenarioDiagnosticFailures);

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
    const cpu = runOptions.runCPUOracle ? runMatchedCPUOracle(scenario, grid, oracleSteps, runOptions) : undefined;
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
      performanceProfile: runOptions.performanceProfile,
      passedDiagnostics: runOptions.performanceProfile ? undefined : scenarioDiagnosticFailures.length === 0,
      qualityGates: runOptions.performanceProfile
        ? "final-authority-only"
        : "evaluated" }));
  }
} finally {
  Reflect.deleteProperty(globalThis, "navigator");
}

if (failures.length > 0) throw new Error(`WebGPU smoke diagnostic failures:\n${failures.map((failure) => `- ${failure}`).join("\n")}`);
