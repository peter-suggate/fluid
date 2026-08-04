import { pathToFileURL } from "node:url";
import { EulerianFluidSolver } from "../lib/eulerian-solver";
import type { GPUSolverInstance, SimulationMethod } from "../lib/methods/types";
import { octreeMethod } from "../lib/methods/octree";
import { initializeRigidBodies } from "../lib/rigid-body";
import type { SceneDescription } from "../lib/model";
import { createSingleTallCellProbeControlLayout, createSingleTallCellProbeLayout, type SingleTallCellProbeOptions } from "../lib/tall-cell-grid";
import type { GPUEulerianInfo, GPUQuality } from "../lib/webgpu-eulerian";
import { summarizeDriftOscillation } from "../lib/tall-cell-diagnostics";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import { FINE_LEVELSET_VOLUME_VALID, unpackFineLevelSetGPUVolumeControl }
  from "../lib/webgpu-octree-fine-levelset-volume";
import { OCTREE_LOSASSO_COARSE_PHI_MAGIC }
  from "../lib/webgpu-octree-losasso-coarse-phi.wgsl";
import { decodeFineLevelSetRecurringRejectionClauses, FINE_LEVELSET_TOPOLOGY_ERROR,
  unpackFineLevelSetGPUTopologyControl }
  from "../lib/webgpu-octree-fine-levelset-topology";
import { unpackFineToCoarseGPUControl }
  from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { unpackOctreePowerCoarseLevelSetControl }
  from "../lib/webgpu-octree-power-coarse-levelset";
import { auditSection5FineRestriction }
  from "../lib/power-liquids-restriction-audit";
import { octreePowerHybridWorkVerdict }
  from "../lib/octree-power-hybrid";
import { readFineLevelSetWorksetHeader } from "../lib/octree-fine-levelset-bricks";
import { decodeStructuredProjectionEnergy }
  from "../lib/webgpu-octree-structured-dynamics";
import {
  passBrokerBoundaryAuditSnapshot,
  resetPassBrokerBoundaryAuditTotals,
  type PassBrokerBoundaryAuditSnapshot,
} from "../lib/webgpu-pass-broker";
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
  GPUResidentMemoryCensus,
  auditCommandEncoder,
  formatResidentMemoryReport,
  writtenByteLength,
  type GPUCommandAuditReport,
  type GPUFineTimestampBucket,
  type GPUFineTimestampReport,
  type GPUPassTimestampReport,
} from "./webgpu-smoke-gpu-audits";
import {
  formatInitializationCensus,
  readInitializationCensus,
} from "../lib/gpu-initialization";
import {
  gravitationalPotentialEnergyProxy,
  initialSeedBrickBounds,
  inspectColumnBases,
  inspectTallVolumeGaps,
  readBufferBinding,
  readBufferBindingsPacked,
  readCompactOctreePressureState3D,
  readCompactOctreeVelocityField3D,
  readCubicVolumeField as readCubicVolumeFieldSnapshot,
  readFloatTexture2D,
  readFloatTexture3D,
  readFineUpperSurfaceField,
  readFinePhiSymmetrySource,
  readFluidBrickSnapshot,
  readGlobalFineGenerationDiagnostics,
  readSparseVoxelStats,
  readTallVelocityField3D,
  readTallVelocityTexture3D,
  readVelocityTexture3D,
  smokeRenderHybridPresentation,
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
  createSmokeScenario,
  isSmokeScenarioId,
  smokeScenarioIds,
  summarizeScalarField,
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

function exactFieldFingerprint(field: ArrayLike<number>): Readonly<{
  length: number; hashA: string; hashB: string;
}> {
  const scratch = new ArrayBuffer(4), scratchFloat = new Float32Array(scratch);
  const scratchWord = new Uint32Array(scratch);
  const floatWords = field instanceof Float32Array
    ? new Uint32Array(field.buffer, field.byteOffset, field.length) : undefined;
  const integerWords = field instanceof Uint32Array ? field : undefined;
  let a = 0x811c9dc5, b = 0x9e3779b9;
  for (let index = 0; index < field.length; index += 1) {
    let word: number;
    if (floatWords) word = floatWords[index]!;
    else if (integerWords) word = integerWords[index]!;
    else { scratchFloat[0] = Number(field[index]); word = scratchWord[0]!; }
    a = Math.imul(a ^ word, 0x01000193) >>> 0;
    b = (Math.imul(b ^ (word + index), 0x85ebca6b) + 0xc2b2ae35) >>> 0;
  }
  return Object.freeze({ length: field.length,
    hashA: a.toString(16).padStart(8, "0"), hashB: b.toString(16).padStart(8, "0") });
}

function losassoRasterCutoverFailures(
  raster: HybridPresentationSmokeStats,
  fineGeneration: number,
): readonly string[] {
  const failures: string[] = [];
  const reverse = raster.reverseView;
  if (raster.frontInterfacePixels === 0 || raster.backInterfacePixels === 0
    || raster.pairedInterfacePixels === 0 || raster.backOnlyInterfacePixels !== 0) {
    failures.push("forward raster lacks closed paired front/back interface coverage");
  }
  if (!reverse || reverse.frontInterfacePixels === 0 || reverse.backInterfacePixels === 0
    || reverse.pairedInterfacePixels === 0 || reverse.backOnlyInterfacePixels !== 0) {
    failures.push("reverse raster lacks closed paired front/back interface coverage");
  }
  const holeCount = raster.enclosedSurfaceHoles.front.count
    + raster.enclosedSurfaceHoles.back.count
    + (reverse?.enclosedSurfaceHoles.front.count ?? 0)
    + (reverse?.enclosedSurfaceHoles.back.count ?? 0);
  const slitCount = raster.narrowVerticalSlits.count
    + (reverse?.narrowVerticalSlits.count ?? 0);
  if (holeCount !== 0) failures.push(`raster contains ${holeCount} enclosed surface holes`);
  if (slitCount !== 0) failures.push(`raster contains ${slitCount} narrow vertical slits`);
  if (raster.rendererValidationErrorCount !== 0 || raster.rendererUncapturedErrorCount !== 0) {
    failures.push("renderer reported validation or uncaptured GPU errors");
  }
  if (raster.surfaceGeometrySource !== "global-fine-coarse"
    || raster.globalFineCrossingPublished !== true
    || raster.presentationFallbackActive !== false
    || (raster.globalFineAuthorityLatch ?? 0) === 0) {
    failures.push("renderer did not consume the current production global-fine/coarse geometry");
  }
  if ((raster.vertexCount ?? 0) === 0 || (raster.activeCubeCount ?? 0) === 0
    || raster.vertexAllocator !== raster.vertexCount
    || (raster.vertexCount ?? Infinity) > (raster.vertexCapacity ?? -1)
    || (raster.activeCubeCount ?? Infinity) > (raster.activeCubeCapacity ?? -1)) {
    failures.push("production surface geometry is empty or its allocation receipt is invalid");
  }
  const retained = raster.globalFineAuthorityTransition;
  if (!retained || retained.cleanFineCoarseRequired !== true
    || retained.validGeneration !== fineGeneration
    || retained.unpublishedGeneration !== fineGeneration + 1
    || retained.retainedGeometrySource !== "retained-previous"
    || retained.retainedFrontInterfacePixels !== raster.frontInterfacePixels
    || retained.retainedBackInterfacePixels !== raster.backInterfacePixels
    || retained.retainedFrontInterfaceHash !== raster.frontInterfaceHash
    || retained.retainedBackInterfaceHash !== raster.backInterfaceHash) {
    failures.push("unpublished-generation probe did not coherently retain the accepted raster");
  }
  return Object.freeze(failures);
}
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

const availableMethods = [octreeMethod];
const methodFilter = process.env.FLUID_METHOD?.split(",").map((value) => value.trim()).filter(Boolean);
const methods = availableMethods.filter((method) => !methodFilter || methodFilter.includes(method.id));
if (methods.length === 0 || (methodFilter && methodFilter.length !== methods.length)) throw new Error(`Unknown FLUID_METHOD=${process.env.FLUID_METHOD}; expected octree`);

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
  rasterInitialFinal: boolean;
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
    rasterInitialFinal: collect.raster === "initial-final",
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
/** An attached Metal recording does not publish encoder metadata until it has
 * observed GPU work from the target process. Prime that stream while the real
 * first advance is still gated so its isolated Losasso pass labels survive. */
const firstAdvanceProfileMetadataWarm = firstAdvanceProfileGate
  ? new Promise<void>((resolve) => {
    process.once("SIGUSR2", resolve);
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
const octreeCoarseBackendOverride = process.env.FLUID_COARSE_BACKEND;
if (octreeCoarseBackendOverride !== undefined
  && octreeCoarseBackendOverride !== "losasso"
  && octreeCoarseBackendOverride !== "power2017") {
  throw new Error("FLUID_COARSE_BACKEND must be losasso or power2017");
}
const octreeTopologyCadenceOverride = process.env.FLUID_TOPOLOGY_CADENCE_ADVANCES === undefined
  ? undefined : Number(process.env.FLUID_TOPOLOGY_CADENCE_ADVANCES);
if (octreeTopologyCadenceOverride !== undefined
  && (!Number.isSafeInteger(octreeTopologyCadenceOverride)
    || octreeTopologyCadenceOverride < 1 || octreeTopologyCadenceOverride > 8)) {
  throw new Error("FLUID_TOPOLOGY_CADENCE_ADVANCES must be an integer from 1 through 8");
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
/** Identity marker for the production-visible factor-4 D4 cutover command.
 * Generic Losasso lanes still receive the native authority oracle, while this
 * marker additionally refuses any drift in the named release gate's wiring. */
const losassoD4CutoverRequested = process.env.FLUID_LOSASSO_D4_CUTOVER === "1";
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
 * downgrades named tripwires to loud warnings for triage only.
 *
 * `FLUID_TRIPWIRES=failfast` adds a per-step queue fence, decode and verdict so
 * the run dies at the step that tripped instead of at end-of-run. Detection is
 * identical either way -- the same bytes, the same predicates, the same fatal
 * outcome -- so this is a *diagnostic* mode, not a stricter one:
 *
 *  - On a red run it is decisive. The large dam reported 732 trips across four
 *    ids at end-of-run; under fail-fast it reports one `air-support-failure` at
 *    step 247 and the other 731 are revealed as downstream absorption.
 *  - On a green run it buys nothing at all -- there is no first trip to stop at
 *    -- and costs +26.8% wall on the large lane, because fencing every step
 *    removes host/GPU overlap. A throughput number taken under it is not
 *    comparable to one taken without it.
 *
 * So: diagnosis and correctness lanes want `failfast`; throughput lanes want
 * `1`, which still fails the run on any trip, just at the end of it. `0`
 * remains the only mode that stops gating, and benchmark lanes refuse it. */
const TRIPWIRE_IDS = ["topology-rollback", "restriction-unaccepted",
  "mgpcg-nonconvergence", "fine-band-sentinel", "air-support-failure"] as const;
type TripwireId = (typeof TRIPWIRE_IDS)[number];
const tripwireMode = process.env.FLUID_TRIPWIRES;
if (tripwireMode !== undefined && tripwireMode !== "0" && tripwireMode !== "1"
  && tripwireMode !== "failfast") {
  throw new Error("FLUID_TRIPWIRES must be 0, 1, or failfast");
}
const tripwiresDisabled = tripwireMode === "0";
/** Terminate at the first trip rather than after the measured window. Carries a
 * per-step queue fence, so it is never on by default: see the mode note above. */
const tripwiresFailFast = tripwireMode === "failfast";
const tripwiresForcedRequired = tripwireMode === "1" || tripwiresFailFast;
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
  /** Which pass-boundary CAUSES produced the frame's compute passes, summed
   * over every `PassBroker` this process constructed after warmup. Unlike the
   * command audit this needs no device proxying, so it is always present. */
  gpuPassBoundaryAudit?: PassBrokerBoundaryAuditSnapshot;
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
    gpuPassBoundaryAudit: result.gpuPassBoundaryAudit,
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
    structuredAirSupportRows: info.structuredAirSupportRows,
    structuredAirSupportCells: info.structuredAirSupportCells,
    structuredAirSupportCapacity: info.structuredAirSupportCapacity,
    structuredAirSupportFaceItems: info.structuredAirSupportFaceItems,
    structuredAirSupportSeedFaces: info.structuredAirSupportSeedFaces,
    structuredAirSupportMarchDepth: info.structuredAirSupportMarchDepth,
    structuredAirSupportFailureFlags: info.structuredAirSupportFailureFlags,
    structuredAirSupportFailureItem: info.structuredAirSupportFailureItem,
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
  const rasterInitialFinalRequested = options.rasterInitialFinal;
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
  // Resident footprint, not per-advance allocation. Off by default because it
  // captures a stack per `createBuffer` to attribute the bytes to a module; the
  // cost is bounded by construction-time allocation and is invisible in the
  // measured window, but a wall being compared should carry no unasked-for work.
  const residentMemoryCensus = (process.env.FLUID_GPU_MEMORY_CENSUS ?? "") !== ""
    ? new GPUResidentMemoryCensus() : undefined;
  const initCensusEnabled = process.env.FLUID_GPU_INIT_CENSUS === "1";
  let createBufferElapsedMs = 0, createBufferCalls = 0, createBufferBytes = 0;
  // Construction is paid once per benchmark arm and every arm is a fresh
  // process, so it lands inside the A/B loop's wall clock in full. Report it on
  // process exit rather than after construction: the runs worth profiling for
  // startup cost include the ones that never reach a measured window, and a
  // census that only prints on the happy path cannot diagnose a red lane.
  if (initCensusEnabled) {
    process.on("exit", () => {
      const entries = readInitializationCensus();
      if (entries.length === 0) return;
      console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
        phase: "gpu-init", entries,
        createBuffer: { calls: createBufferCalls, bytes: createBufferBytes,
          elapsed_ms: createBufferElapsedMs } }));
      console.log("--- GPU initialization census ---");
      console.log(formatInitializationCensus(entries));
      console.log(`  createBuffer: ${createBufferCalls} calls,`
        + ` ${(createBufferBytes / (1024 * 1024)).toFixed(1)} MiB,`
        + ` ${createBufferElapsedMs.toFixed(1)} ms`
        + ` (${(createBufferBytes / (1024 * 1024) / (createBufferElapsedMs / 1000)).toFixed(0)} MiB/s)`);
    });
  }
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
        const createStarted = initCensusEnabled ? performance.now() : 0;
        const buffer = target.createBuffer(descriptor);
        if (initCensusEnabled) {
          createBufferElapsedMs += performance.now() - createStarted;
          createBufferCalls += 1;
          createBufferBytes += Number(descriptor.size);
        }
        dataFlowAudit?.registry.recordBuffer(buffer, descriptor);
        residentMemoryCensus?.recordBuffer(descriptor, buffer);
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
  if (method.id === "octree" && octreeCoarseBackendOverride !== undefined) {
    values.coarseBackend = octreeCoarseBackendOverride;
  }
  if (method.id === "octree" && octreeTopologyCadenceOverride !== undefined) {
    values.topologyCadenceAdvances = octreeTopologyCadenceOverride;
  }
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
  const losassoCutoverLane = method.id === "octree" && values.coarseBackend === "losasso";
  // Factor one transports phi directly on the accepted octree rows and owns
  // no separate sparse fine-band publication.
  const hasSeparateFineLevelSetBand = method.id === "octree"
    && Number(values.globalFineLevelSetFactor) !== 1;
  const verifyGlobalFineGenerationTransition = globalFineGenerationTransitionRequested
    && hasSeparateFineLevelSetBand;
  if (losassoD4CutoverRequested) {
    const wiringFailures = [
      ...(scenarioId === "symmetric-expansion" ? [] : [`scene=${scenarioId}`]),
      ...(scenario.lane.id === "fine-factor-4" ? [] : [`lane=${scenario.lane.id}`]),
      ...(method.id === "octree" ? [] : [`method=${method.id}`]),
      ...(values.coarseBackend === "losasso" ? [] : [`backend=${String(values.coarseBackend)}`]),
      ...(Number(values.maximumLeafSize) === 16 ? [] : [`maximumLeafSize=${String(values.maximumLeafSize)}`]),
      ...(Number(values.interfaceRefinementBandCells) === 4
        ? [] : [`interfaceBand=${String(values.interfaceRefinementBandCells)}`]),
      ...(Number(values.globalFineLevelSetFactor) === 4
        ? [] : [`fineFactor=${String(values.globalFineLevelSetFactor)}`]),
      ...(rasterInitialFinalRequested ? [] : ["initial-final-raster=off"]),
      ...(rasterCheckpointRequested ? [] : ["checkpoint-raster=off"]),
      ...(verifyGlobalFineGenerationTransition ? [] : ["retained-generation-probe=off"]),
      ...(process.env.FLUID_RASTER_MESH_SYMMETRY === "1" ? [] : ["exact-D4-raster=off"]),
    ];
    if (wiringFailures.length > 0) {
      throw new Error(`Losasso D4 cutover gate wiring drifted: ${wiringFailures.join(", ")}`);
    }
  }
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
    solver = method.id === "octree" && method.createSolverAsync
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
  const initialGlobalFineRaster = rasterInitialFinalRequested && method.id === "octree"
    ? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies,
      verifyGlobalFineGenerationTransition)
    : undefined;
  if (initialGlobalFineRaster) {
    // Emit the pre-step renderer evidence immediately.  A later simulation
    // transaction may deliberately reject and roll back, but that must not
    // hide whether reset-time global-fine rasterization was already visible.
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "initial-global-fine-raster", ...initialGlobalFineRaster }));
    if (process.env.FLUID_RASTER_MESH_SYMMETRY === "1") {
      const mesh = initialGlobalFineRaster.surfaceMeshSymmetry;
      const phi = initialGlobalFineRaster.finePhiSymmetry;
      const sharp = initialGlobalFineRaster.sharpPatchRaster;
      // Factor one has no separate fine publication. Its fixed tetrahedral
      // diagonal can produce reflection-dependent floating-point normals even
      // when the geometric vertex set is exactly symmetric, so use position
      // coverage for the missing-face oracle in that mode.
      if (!mesh || !sharp || sharp.patchCount === 0 || sharp.invalidPatchCount !== 0
        || (hasSeparateFineLevelSetBand
          ? mesh.exactMismatchCount !== 0 : mesh.exactPositionMismatchCount !== 0)
        || mesh.nonFiniteCount !== 0
        || (hasSeparateFineLevelSetBand && !phi)
        || (phi && (phi.supportMismatchCount !== 0 || phi.exactValueMismatchCount !== 0
          || phi.nonFiniteCount !== 0 || phi.maximumAbsoluteError !== 0))) {
        throw new Error(`initial global-fine raster oracle rejected: ${JSON.stringify({ mesh, phi, sharp })}`);
      }
    }
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
  if (firstAdvanceProfileGateReleased && firstAdvanceProfileMetadataWarm) {
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "before-first-advance", profileGate: "waiting-for-sigusr2" }));
    await firstAdvanceProfileMetadataWarm;
    const metadataWarmEncoder = instrumentedDevice.createCommandEncoder({
      label: "Profile encoder metadata warmup",
    });
    const metadataWarmPass = metadataWarmEncoder.beginComputePass({
      label: "Profile encoder metadata warmup",
    });
    metadataWarmPass.end();
    instrumentedDevice.queue.submit([metadataWarmEncoder.finish()]);
    await instrumentedDevice.queue.onSubmittedWorkDone();
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "before-first-advance", profileGate: "metadata-warm" }));
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
  // The same boundary is exactly where the resident footprint is worth reading:
  // the solver is constructed and t=0 is published, so what it holds now is what
  // it holds for the run. Emitted before the reset discards the only record of it.
  if (residentMemoryCensus) {
    const residentMemory = residentMemoryCensus.snapshot();
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "gpu-memory", when: "after-construction", residentMemory }));
    console.log(formatResidentMemoryReport(residentMemory));
  }
  // Same window, same reason: the pass-boundary census is process-wide and
  // cumulative from module load, so without this the per-advance numbers carry
  // construction plus the cold bootstrap encoders
  // (encodeColdTopologySignatureBaseline, encodeAnalyticBootstrap). Reset is
  // unconditional because the census costs nothing to keep.
  resetPassBrokerBoundaryAuditTotals();
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
  const powerGenerationAuditCapacity = method.id === "octree" && !losassoCutoverLane
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
  // transaction's unaccepted-row count, the MGPCG converged word, the fine
  // worklist header, and the air-support first-failure receipt. They are
  // copied GPU-side by a tiny encoder that
  // rides after the advance's own submission and is mapped exactly once,
  // after the measured window closes; nothing here drains the pipeline.
  const TRIPWIRE_RECORD = Object.freeze({
    // The full 64-byte topology control: words 10..15 are the sticky
    // first-rejection latch (reason bits, first rejected generation, and the
    // producer's own pre-finalize flags/item) that per-generation clears
    // never touch.
    topologyOffsetBytes: 0, topologyBytes: 64,
    restrictionOffsetBytes: 64, restrictionBytes: 32,
    mgpcgOffsetBytes: 96, mgpcgBytes: 64,
    coarseOffsetBytes: 160, coarseBytes: 64,
    fineHeaderOffsetBytes: 224, fineHeaderBytes: 28,
    // Air-support scratch[0..1] is the live verdict and [38..39] is the first
    // rejected preceding transaction. Fine-transport governor words 0..7 and
    // 46..56 carry the schedule and sleep/repair cause for the same step.
    airSupportOffsetBytes: 252, airSupportBytes: 16,
    transportScheduleOffsetBytes: 268, transportScheduleBytes: 32,
    transportSleepOffsetBytes: 300, transportSleepBytes: 44,
    // The air-support carrier-free ledger (scratch 41/42) and the forensic
    // record the producer expands it into (scratch 74..95): the failing
    // owned-face item with its resolved cell geometry, and the per-axis
    // counts of patches the march could not reach. Without these an
    // `air-support-failure` trip names no face at all.
    airSupportLedgerOffsetBytes: 344, airSupportLedgerBytes: 8,
    airSupportForensicOffsetBytes: 352, airSupportForensicBytes: 88,
    strideBytes: 440,
  });
  /** The benchmark and acceptance lanes must evaluate every tripwire. Any
   * other octree run captures them opportunistically: a trip still fails, but
   * a scene with no compact fine authority is "not applicable" rather than a
   * wiring failure. */
  const tripwiresRequired = !tripwiresDisabled && method.id === "octree" && !losassoCutoverLane
    && (tripwiresForcedRequired || powerGenerationAuditRequested || performanceProfileRequested);
  const tripwireCapacity = !tripwiresDisabled && method.id === "octree" && !losassoCutoverLane
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
  /** Live per-step readback of the record just written into the ring.
   *
   * We NEVER self-heal from a tripped counter, so a trip must end the run at
   * the step that produced it: a frozen fine band that keeps being stepped for
   * another 180 advances is 180 advances of measured garbage. The ring stays
   * the whole-run forensics source; this single-record buffer is what makes the
   * verdict available *before* the next advance is submitted.
   *
   * Map/unmap cycle (why this is correct):
   *  - the buffer is COPY_DST|MAP_READ and is written only by the tripwire
   *    encoder, which is submitted while the buffer is unmapped;
   *  - `mapAsync` resolves only after every previously submitted use of the
   *    buffer has completed, so the mapped bytes are exactly this step's copy
   *    (and, transitively, this step's post-advance control state, because the
   *    copy encoder is submitted after the advance on the same queue);
   *  - the host copies the record out and `unmap()`s in a `finally` before the
   *    loop can reach the next capture, so no encoder ever references a mapped
   *    buffer and no second `mapAsync` overlaps a pending one. */
  const tripwireLiveReadback = tripwireSnapshot && tripwiresFailFast
    ? device.createBuffer({
      label: "Live silent-failure tripwire record",
      size: TRIPWIRE_RECORD.strideBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    : undefined;
  /** Host-owned copy of the mapped record; every offset in TRIPWIRE_RECORD is
   * 4-byte aligned, so `Uint32Array` views over it are always legal. */
  const tripwireLiveRecord = new Uint8Array(TRIPWIRE_RECORD.strideBytes);
  type TripwireTrip = {
    id: TripwireId; step: number; fineGeneration: number; detail: Record<string, unknown>;
  };
  /** Allow-listed trips observed live. They do not stop the run, but when a
   * later step dies they are the chain that led to it, and the end-of-run walk
   * that normally reports them never gets to run. */
  const tripwireLiveAllowed: TripwireTrip[] = [];
  let tripwireLiveFence_ms = 0, tripwireLiveDecode_ms = 0;
  if (tripwiresDisabled) {
    console.error("[tripwires] DISABLED by FLUID_TRIPWIRES=0: topology rollback,"
      + " unaccepted restriction rows, MGPCG non-convergence and the fine-band"
      + " capacity sentinel, and air-support publication failure are NOT gated in this run");
  }
  if (tripwireAllowList.size !== 0) {
    console.error(`[tripwires] downgraded to warnings by FLUID_TRIPWIRE_ALLOW: ${
      Array.from(tripwireAllowList).join(", ")}`);
  }
  /** Live control buffers the tripwire record is copied from.
   *
   * The authoritative global-fine source already carries the topology control
   * selected by `globalFinePublishedIsA`. Read it directly from that production
   * source; hierarchical work-accounting buffers were retired and must not be
   * resurrected merely to feed this QA snapshot.
   * A missing source is a hard failure on a required lane, never a silent
   * skip -- an unevaluable tripwire is what this work item exists to kill. */
  const tripwireSources = () => {
    const authority = solver as GPUSolverInstance & {
      mgpcgControl?: GPUBuffer;
      globalFineRestrictionControl?: GPUBuffer;
      globalFineSummaryDebug?: { coarseControl: GPUBuffer };
      airSupportScratch?: GPUBuffer;
      workAccountingBuffers?: { fineTransportGovernor?: GPUBufferBinding };
    };
    const topologyControl = authority.globalFineLevelSetSource?.topologyControl;
    const topology: GPUBufferBinding | undefined = topologyControl
      ? { buffer: topologyControl, offset: 0, size: topologyControl.size }
      : undefined;
    return {
      topology,
      restriction: authority.globalFineRestrictionControl,
      mgpcg: authority.mgpcgControl,
      coarse: authority.globalFineSummaryDebug?.coarseControl,
      fineWorklist: authority.globalFineLevelSetSource?.worklist,
      airSupport: authority.airSupportScratch,
      transportGovernor: authority.workAccountingBuffers?.fineTransportGovernor,
    };
  };
  /** Encode one complete tripwire record into `destination` at `base`.
   *
   * The ring and the live single-record buffer are filled by two calls to this
   * one writer inside the same encoder, so they cannot drift: a copy added for
   * one is a copy added for both. Every source has already been proven present
   * and wide enough by the caller. */
  const encodeTripwireRecordCopies = (
    encoder: GPUCommandEncoder, sources: ReturnType<typeof tripwireSources>,
    destination: GPUBuffer, base: number,
  ): void => {
    encoder.copyBufferToBuffer(sources.topology!.buffer, sources.topology!.offset ?? 0,
      destination, base + TRIPWIRE_RECORD.topologyOffsetBytes,
      TRIPWIRE_RECORD.topologyBytes);
    encoder.copyBufferToBuffer(sources.restriction!, 0, destination,
      base + TRIPWIRE_RECORD.restrictionOffsetBytes, TRIPWIRE_RECORD.restrictionBytes);
    encoder.copyBufferToBuffer(sources.mgpcg!, 0, destination,
      base + TRIPWIRE_RECORD.mgpcgOffsetBytes, TRIPWIRE_RECORD.mgpcgBytes);
    encoder.copyBufferToBuffer(sources.coarse!, 0, destination,
      base + TRIPWIRE_RECORD.coarseOffsetBytes, TRIPWIRE_RECORD.coarseBytes);
    encoder.copyBufferToBuffer(sources.fineWorklist!, 0, destination,
      base + TRIPWIRE_RECORD.fineHeaderOffsetBytes, TRIPWIRE_RECORD.fineHeaderBytes);
    encoder.copyBufferToBuffer(sources.airSupport!, 0, destination,
      base + TRIPWIRE_RECORD.airSupportOffsetBytes, 8);
    encoder.copyBufferToBuffer(sources.airSupport!, 38 * 4, destination,
      base + TRIPWIRE_RECORD.airSupportOffsetBytes + 8, 8);
    encoder.copyBufferToBuffer(sources.airSupport!, 41 * 4, destination,
      base + TRIPWIRE_RECORD.airSupportLedgerOffsetBytes,
      TRIPWIRE_RECORD.airSupportLedgerBytes);
    encoder.copyBufferToBuffer(sources.airSupport!, 74 * 4, destination,
      base + TRIPWIRE_RECORD.airSupportForensicOffsetBytes,
      TRIPWIRE_RECORD.airSupportForensicBytes);
    encoder.copyBufferToBuffer(sources.transportGovernor!.buffer,
      sources.transportGovernor!.offset ?? 0, destination,
      base + TRIPWIRE_RECORD.transportScheduleOffsetBytes,
      TRIPWIRE_RECORD.transportScheduleBytes);
    encoder.copyBufferToBuffer(sources.transportGovernor!.buffer,
      (sources.transportGovernor!.offset ?? 0) + 46 * 4, destination,
      base + TRIPWIRE_RECORD.transportSleepOffsetBytes,
      TRIPWIRE_RECORD.transportSleepBytes);
  };
  /** Decode and evaluate ONE captured record.
   *
   * The live per-step path and the end-of-run ring walk both call this on
   * byte-identical records, so a trip can never be seen by one and missed by
   * the other. `emitTopologyTrace` belongs to the live path alone: the trace
   * has to survive the step that dies, and emitting it from both walks would
   * duplicate every record on a run that reaches the end. */
  const evaluateTripwireRecord = (
    words: (offsetBytes: number, byteLength: number) => Uint32Array,
    step: number, fineGeneration: number, emitTopologyTrace: boolean,
  ): TripwireTrip[] => {
    const trips: TripwireTrip[] = [];
    const trip = (id: TripwireId, detail: Record<string, unknown>) =>
      trips.push({ id, step, fineGeneration, detail });
    const header = readFineLevelSetWorksetHeader(words(
      TRIPWIRE_RECORD.fineHeaderOffsetBytes, TRIPWIRE_RECORD.fineHeaderBytes));
    const recurringRejection = header?.generation === 0xffff_ffff
      && header.activeCount === 0xffff_ffff ? {
        clauseMask: header.capacity,
        clauses: decodeFineLevelSetRecurringRejectionClauses(header.capacity),
        unknownClauseMask: (header.capacity & ~0x1fff) >>> 0,
        landingDisplacementBits: header.flags,
      } : undefined;
    const airSupport = words(TRIPWIRE_RECORD.airSupportOffsetBytes,
      TRIPWIRE_RECORD.airSupportBytes);
    const transportSchedule = Array.from(words(
      TRIPWIRE_RECORD.transportScheduleOffsetBytes,
      TRIPWIRE_RECORD.transportScheduleBytes));
    const transportSleep = Array.from(words(
      TRIPWIRE_RECORD.transportSleepOffsetBytes,
      TRIPWIRE_RECORD.transportSleepBytes));
    const fatalChainForensics = {
      airSupportLive: { flags: airSupport[0], firstError: airSupport[1] },
      airSupportLatched: { flags: airSupport[2], firstError: airSupport[3] },
      recurringRejection,
      transportSchedule,
      transportSleep,
    };
    // 1. Topology rollback. `settleFinePublication` returns early on the
    //    clean path and writes control[5]=1 only on the rollback branch;
    //    the host re-zeroes control words 0..7 at the start of every
    //    topology encode, so a set word is always this generation's own
    //    verdict, never a stale latch.
    const topologyControl = words(TRIPWIRE_RECORD.topologyOffsetBytes,
      TRIPWIRE_RECORD.topologyBytes);
    const topology = unpackFineLevelSetGPUTopologyControl(topologyControl);
    /** The producer's carrier-free record, decoded. `carrierFree` is absent
     * unless the no-carrier ledger actually fired; `marchUnreachedPatchesByAxis`
     * is published every step and is non-zero whenever the demand corridor was
     * disconnected and the exhaustive closest-face transform had to resolve it,
     * which is a corridor-coverage signal even on a step that does not fail. */
    const airSupportCarrierForensics = () => {
      const ledger = words(TRIPWIRE_RECORD.airSupportLedgerOffsetBytes,
        TRIPWIRE_RECORD.airSupportLedgerBytes);
      const forensic = words(TRIPWIRE_RECORD.airSupportForensicOffsetBytes,
        TRIPWIRE_RECORD.airSupportForensicBytes);
      return {
        marchUnreachedPatchesByAxis: [forensic[12], forensic[13], forensic[14]],
        uncarriedPatchesByAxis: [forensic[15], forensic[16], forensic[17]],
        supportRows: forensic[21],
        ...((ledger[0] ?? 0) === 0 ? {} : {
          carrierFree: {
            occurrences: ledger[0], item: ledger[1], faceRow: forensic[0],
            origin: [forensic[1], forensic[2], forensic[3]],
            size: forensic[4], caseId: forensic[5],
            transform: (forensic[6] ?? 0) & 63,
            recordFlags: ((forensic[6] ?? 0) >>> 6) & 0xff,
            axis: Math.floor((forensic[7] ?? 0) / 4), quadrant: (forensic[7] ?? 0) % 4,
            negativeRow: forensic[8], directRows: forensic[9],
            incidentRows: forensic[10], positiveRow: forensic[11],
          },
        }),
      };
    };
    if (emitTopologyTrace && process.env.FLUID_FINE_TOPOLOGY_TRACE === "1") {
      // stderr: the benchmark harness pipes child stdout through an
      // NDJSON filter that would drop this record type.
      console.error(JSON.stringify({ record: "fine-topology-trace", step,
        airSupport: airSupportCarrierForensics(),
        fineGeneration, interfaceBricks: topology.interfaceBricks,
        interfaceSeedBricks: topology.interfaceSeedBricks,
        desiredBricks: topology.desiredBricks,
        activatedBricks: topology.activatedBricks,
        rolledBack: topology.rolledBack, flags: topology.flags,
        reason: topology.downstreamFinalizeReason,
        requiredDesiredBricks: topology.requiredDesiredBricks,
        requiredExact: topology.requiredDesiredBricksExact,
        control: Array.from(topologyControl) }));
    }
    if (topology.rolledBack) {
      trip("topology-rollback", { rolledBack: true,
        flags: topology.flags, published: topology.published,
        downstreamFinalizeReason: topology.downstreamFinalizeReason,
        interfaceBricks: topology.interfaceBricks,
        desiredBricks: topology.desiredBricks,
        activatedBricks: topology.activatedBricks,
        control: Array.from(topologyControl), fatalChainForensics });
    }
    // 2. Section 5 restriction authority. Aanjaneya et al. deliberately
    //    keep the fine SPGrid only around the surface; the background
    //    octree owns every other row. A global uncovered-row fraction is
    //    therefore scene-dependent and invalid (the ocean legitimately
    //    leaves about 91% of coarse rows outside the fine band). Audit
    //    the real two-authority receipt instead: every accepted fine row
    //    becomes one coarse correction and those rows cover the complete
    //    interface set.
    const restrictionWords = words(TRIPWIRE_RECORD.restrictionOffsetBytes,
      TRIPWIRE_RECORD.restrictionBytes);
    const restriction = unpackFineToCoarseGPUControl(restrictionWords);
    const coarseWords = words(TRIPWIRE_RECORD.coarseOffsetBytes,
      TRIPWIRE_RECORD.coarseBytes);
    const coarse = unpackOctreePowerCoarseLevelSetControl(coarseWords);
    const restrictionAudit = auditSection5FineRestriction(restriction, coarse);
    if (restrictionAudit.failure && hasSeparateFineLevelSetBand) {
      trip("restriction-unaccepted", { reason: restrictionAudit.failure,
        ...restrictionAudit, restrictionControl: Array.from(restrictionWords),
        coarseControl: Array.from(coarseWords), fatalChainForensics });
    }
    // 3. Solver convergence. Non-convergence at the encoded budget
    //    publishes the SEED pressure and fails nothing today; this is the
    //    guard for that cliff. Steps that executed no iterations are
    //    exempt by construction (nothing to converge); a terminal count of
    //    zero is gated per run by tools/benchmark-power-dam.ts.
    const mgpcgWords = words(TRIPWIRE_RECORD.mgpcgOffsetBytes,
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
    if (header === undefined) {
      trip("fine-band-sentinel", { unevaluable: true,
        reason: "fine worklist header could not be decoded", fatalChainForensics });
    } else if (header.activeCount === 0xffff_ffff) {
      trip("fine-band-sentinel", { activeCount: header.activeCount,
        sentinel: "0xFFFFFFFF", capacity: header.capacity,
        generation: header.generation, flags: header.flags,
        recurringRejection, fatalChainForensics });
    } else if ((topology.flags & FINE_LEVELSET_TOPOLOGY_ERROR.capacity) !== 0
      || topology.requiredDesiredBricks > header.capacity) {
      trip("fine-band-sentinel", { activeCount: header.activeCount,
        retainedPublication: topology.rolledBack,
        requiredDesiredBricks: topology.requiredDesiredBricks,
        requiredDesiredBricksExact: topology.requiredDesiredBricksExact,
        capacity: header.capacity, generation: header.generation,
        topologyFlags: topology.flags, fatalChainForensics });
    }
    // 5. Air-support publication. The live scratch verdict and the
    // first preceding failure latch are both copied at this step's queue
    // boundary, so a later transaction cannot erase the originating
    // rejection before the harness sees it.
    const liveAirSupportFailure = (airSupport[0] ?? 0) !== 0;
    const latchedAirSupportFailure = (airSupport[2] ?? 0) !== 0;
    if (liveAirSupportFailure || latchedAirSupportFailure) {
      trip("air-support-failure", { ...airSupportCarrierForensics(), fatalChainForensics });
    }
    return trips;
  };
  /** The single failure report. `where` is the only difference between dying
   * at the step that tripped and the end-of-run walk; every field of the
   * payload -- counts, the first twelve trips with their complete per-tripwire
   * detail and `fatalChainForensics`, and the last trip -- is identical. */
  const tripwireFailure = (failing: readonly TripwireTrip[], where: string): Error => {
    const byId: Record<string, number> = {};
    for (const entry of failing) byId[entry.id] = (byId[entry.id] ?? 0) + 1;
    const allowedById: Record<string, number> = {};
    for (const entry of tripwireLiveAllowed) {
      allowedById[entry.id] = (allowedById[entry.id] ?? 0) + 1;
    }
    return new Error(`silent-failure tripwire(s) tripped ${where}: ${JSON.stringify({
      counts: byId, firstTrips: failing.slice(0, 12),
      lastTrip: failing[failing.length - 1],
      ...(tripwireLiveAllowed.length === 0 ? {} : {
        allowedCounts: allowedById,
        lastAllowedTrip: tripwireLiveAllowed[tripwireLiveAllowed.length - 1],
      }),
    })} (see docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3)`);
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
    const captureCompactPowerStep = method.id === "octree" && !losassoCutoverLane
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
    // Set only when this step actually enqueued a record, so the live readback
    // never maps stale bytes from an earlier step on a lane whose controls are
    // temporarily absent (`tripwiresRequired === false`).
    let capturedThisStep = false;
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
      if (sources.airSupport && sources.airSupport.size < 40 * 4) {
        missing.push(`airSupport (buffer exposes ${sources.airSupport.size} bytes,`
          + " the first-failure latch ends at byte 160)");
      }
      const governorBytes = sources.transportGovernor?.size
        ?? ((sources.transportGovernor?.buffer.size ?? 0) - (sources.transportGovernor?.offset ?? 0));
      if (sources.transportGovernor && governorBytes < 57 * 4) {
        missing.push(`transportGovernor (binding exposes ${governorBytes} bytes,`
          + " the sleep forensics end at byte 228)");
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
        encodeTripwireRecordCopies(encoder, sources, tripwireSnapshot, base);
        // The same record, into the buffer the host maps at this step. One
        // writer, one encoder, one submission: the live verdict and the ring's
        // whole-run forensics are the same bytes by construction. Only fail-fast
        // maps per step, so only fail-fast pays for the second copy.
        if (tripwiresFailFast) {
          encodeTripwireRecordCopies(encoder, sources, tripwireLiveReadback!, 0);
        }
        device.queue.submit([encoder.finish()]);
        tripwireSteps.push(steps);
        tripwireFineGenerations.push(
          (solver as GPUSolverInstance).globalFineLevelSetSource?.generation ?? 0);
        capturedThisStep = true;
      }
      samplingWall_ms += performance.now() - tripwireCaptureStartedAt_ms;
    }
    if (capturedThisStep && tripwiresFailFast) {
      // ---- Fail fast: evaluate step N's tripwires AT step N -----------------
      // We never self-heal. A tripped counter means the physics this harness is
      // measuring is already gone, so the run ends here rather than stepping a
      // frozen solver for another two hundred advances and reporting the trip
      // as an end-of-run footnote.
      //
      // Under `FLUID_TRIPWIRES=1` the ring above still captures this step and
      // the end-of-run walk still fails the run on the same bytes; what is
      // skipped is only the fence that makes the verdict available *now*.
      //
      // `mapAsync` is a queue fence. It is charged like the periodic fence at
      // `completionFenceEverySteps` -- i.e. NOT to samplingWall_ms -- because
      // what it waits for is the solver's own submitted work; subtracting that
      // would make simulationWall_ms smaller than the physics that produced it,
      // which is exactly the fake speedup these tripwires exist to catch. The
      // host-side decode after the fence IS diagnostics and is charged, exactly
      // as the capture above is. Both are reported per run so the cost of
      // fail-fast is measured rather than assumed.
      const fenceStartedAt_ms = performance.now();
      await tripwireLiveReadback!.mapAsync(GPUMapMode.READ, 0, TRIPWIRE_RECORD.strideBytes);
      const decodeStartedAt_ms = performance.now();
      tripwireLiveFence_ms += decodeStartedAt_ms - fenceStartedAt_ms;
      try {
        tripwireLiveRecord.set(new Uint8Array(
          tripwireLiveReadback!.getMappedRange(0, TRIPWIRE_RECORD.strideBytes)));
      } finally {
        // Unmap before anything can encode into this buffer again; the next
        // capture's `copyBufferToBuffer` would be invalid against a mapped one.
        tripwireLiveReadback!.unmap();
      }
      const fineGeneration = tripwireFineGenerations[tripwireFineGenerations.length - 1]!;
      const liveTrips = evaluateTripwireRecord((offsetBytes, byteLength) => new Uint32Array(
        tripwireLiveRecord.buffer, tripwireLiveRecord.byteOffset + offsetBytes, byteLength / 4),
      steps, fineGeneration, true);
      const liveFailing = liveTrips.filter((entry) => !tripwireAllowList.has(entry.id));
      for (const entry of liveTrips) {
        if (tripwireAllowList.has(entry.id)) tripwireLiveAllowed.push(entry);
      }
      const liveDecode_ms = performance.now() - decodeStartedAt_ms;
      tripwireLiveDecode_ms += liveDecode_ms;
      samplingWall_ms += liveDecode_ms;
      if (liveFailing.length !== 0) {
        // FLUID_TRIPWIRE_ALLOW is the only downgrade path, and it has already
        // been applied above. Everything left is fatal at its own step.
        throw tripwireFailure(liveFailing, `at step ${steps}`
          + ` (fine generation ${fineGeneration}) over ${tripwireSteps.length} captured steps`);
      }
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
      const compactPressureState = checkpointSources.has("compact pressure") && method.id === "octree"
        ? await readCompactOctreePressureState3D(device, solver,
          [solver.info.nx, solver.info.ny, solver.info.nz])
        : undefined;
      if (process.env.FLUID_HEAD_DIFFERENTIAL === "1") {
        const fields: Record<string, ArrayLike<number>> = { volume: cubic.field };
        if (compactVelocityField) fields.velocity = compactVelocityField;
        if (compactPressureState) {
          fields.pressure = compactPressureState.pressure;
          fields.rhs = compactPressureState.rhs;
          fields.diagonal = compactPressureState.diagonal;
          fields.topology = compactPressureState.topology;
          if (compactPressureState.section63Diagonal) {
            fields.section63Diagonal = compactPressureState.section63Diagonal;
          }
          if (compactPressureState.section63CaseId) {
            fields.section63CaseId = compactPressureState.section63CaseId;
          }
          if (compactPressureState.section63CoefficientRows) {
            fields.section63Coefficients = compactPressureState.section63CoefficientRows;
            const coefficientRows = compactPressureState.section63CoefficientRows;
            for (let channel = 0; channel < 19; channel += 1) {
              fields[`section63Coefficient${channel}`] = Float32Array.from(
                { length: coefficientRows.length / 19 },
                (_unused, row) => coefficientRows[19 * row + channel]!);
            }
          }
        }
        console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
          phase: "head-differential-fingerprints", step: steps,
          time_s: solver.info.submittedTime_s ?? 0,
          fields: Object.fromEntries(Object.entries(fields)
            .map(([name, field]) => [name, exactFieldFingerprint(field)])) }));
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
        ...(compactPressureState ? {
          pressureField: compactPressureState.pressure,
          pressureRhsField: compactPressureState.rhs,
          pressureDiagonalField: compactPressureState.diagonal,
          ...(compactPressureState.section63Diagonal ? {
            pressureSection63DiagonalField: compactPressureState.section63Diagonal } : {}),
          ...(compactPressureState.section63CaseId ? {
            pressureSection63CaseIdField: compactPressureState.section63CaseId } : {}),
          ...(compactPressureState.initialResidual ? {
            pressureInitialResidualField: compactPressureState.initialResidual } : {}),
          ...(compactPressureState.initialPreconditioned ? {
            pressureInitialPreconditionedField: compactPressureState.initialPreconditioned } : {}),
          ...(compactPressureState.initialPreconditionedImage ? {
            pressureInitialPreconditionedImageField: compactPressureState.initialPreconditionedImage } : {}),
          ...(compactPressureState.preconditionerPreSmoothed ? {
            pressurePreconditionerPreSmoothedField: compactPressureState.preconditionerPreSmoothed } : {}),
          ...(compactPressureState.preconditionerZeroSmoothed ? {
            pressurePreconditionerZeroSmoothedField: compactPressureState.preconditionerZeroSmoothed } : {}),
          ...(compactPressureState.preconditionerFirstOperatorImage ? {
            pressurePreconditionerFirstOperatorImageField: compactPressureState.preconditionerFirstOperatorImage } : {}),
          ...(compactPressureState.preconditionerFirstSmoothed ? {
            pressurePreconditionerFirstSmoothedField: compactPressureState.preconditionerFirstSmoothed } : {}),
          ...(compactPressureState.preconditionerInnerResidual ? {
            pressurePreconditionerInnerResidualField: compactPressureState.preconditionerInnerResidual } : {}),
          ...(compactPressureState.preconditionerInnerCorrection ? {
            pressurePreconditionerInnerCorrectionField: compactPressureState.preconditionerInnerCorrection } : {}),
          ...(compactPressureState.preconditionerPostCorrected ? {
            pressurePreconditionerPostCorrectedField: compactPressureState.preconditionerPostCorrected } : {}),
          topologyField: compactPressureState.topology,
        } : {}),
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
  let losassoCutoverAuthority: Readonly<Record<string, unknown>> | undefined;
  let losassoCutoverRaster: HybridPresentationSmokeStats | undefined;
  if (losassoCutoverLane) {
    const projection = (solver as GPUSolverInstance & { octreeProjection?: {
      readLosassoAuthorityDiagnostics(): Promise<Readonly<{
        authority: readonly number[]; solver: readonly number[]; coarsePhi: readonly number[];
      }> | undefined>;
    } }).octreeProjection;
    const [native, fine, raster] = await Promise.all([
      projection?.readLosassoAuthorityDiagnostics(),
      hasSeparateFineLevelSetBand
        ? readGlobalFineGenerationDiagnostics(device, solver) : Promise.resolve(undefined),
      hasSeparateFineLevelSetBand
        ? smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies, true)
        : Promise.resolve(undefined),
    ]);
    losassoCutoverRaster = raster;
    const authority = native?.authority ?? [], pressure = native?.solver ?? [];
    const coarsePhi = native?.coarsePhi ?? [];
    const failures: string[] = [];
    if (steps < 1) failures.push("no accepted advance executed");
    if (authority.length < 5 || authority[0] === 0 || authority[1] === 0
      || authority[2] === 0 || authority[3] !== 1 || authority[4] !== 0) {
      failures.push("reduced authority is invalid or empty");
    }
    if (pressure.length < 5 || pressure[0] !== 0 || pressure[1] === 0
      || pressure[4] !== authority[1]) {
      failures.push("MGPCG did not converge on the accepted row authority");
    }
    if (coarsePhi.length < 15 || coarsePhi[0] !== OCTREE_LOSASSO_COARSE_PHI_MAGIC
      || coarsePhi[1] !== authority[0] || coarsePhi[2] !== authority[1]
      || coarsePhi[3] !== authority[2] || coarsePhi[13] !== 0) {
      failures.push("coarse phi is invalid or disagrees with the reduced authority");
    }
    if (hasSeparateFineLevelSetBand && (!fine || !fine.publicationValid
      || fine.validSamples === 0 || fine.finiteValidSamples !== fine.validSamples
      || fine.negativeValidSamples === 0 || fine.positiveValidSamples === 0
      || coarsePhi[12] !== fine.generation || coarsePhi[14] !== fine.generation)) {
      failures.push("factor-4 fine phi is invalid, non-finite, one-sided, or generation-incoherent");
    }
    if (hasSeparateFineLevelSetBand) {
      if (!fine || !raster) failures.push("factor-4 production raster evidence is unavailable");
      else failures.push(...losassoRasterCutoverFailures(raster, fine.generation));
    }
    losassoCutoverAuthority = Object.freeze({ backend: "losasso", steps,
      submittedTime_s: solver.info.submittedTime_s, authority, solver: pressure, coarsePhi,
      fine: fine ? { generation: fine.generation, worklistGeneration: fine.worklistGeneration,
        activePages: fine.activePages, capacity: fine.configuredBrickCapacity,
        validSamples: fine.validSamples, finiteValidSamples: fine.finiteValidSamples,
        negativeValidSamples: fine.negativeValidSamples,
        positiveValidSamples: fine.positiveValidSamples,
        publicationValid: fine.publicationValid } : undefined,
      raster: raster ? {
        geometrySource: raster.surfaceGeometrySource,
        frontInterfacePixels: raster.frontInterfacePixels,
        backInterfacePixels: raster.backInterfacePixels,
        pairedInterfacePixels: raster.pairedInterfacePixels,
        backOnlyInterfacePixels: raster.backOnlyInterfacePixels,
        narrowVerticalSlits: raster.narrowVerticalSlits,
        enclosedSurfaceHoles: raster.enclosedSurfaceHoles,
        reverseView: raster.reverseView ? {
          frontInterfacePixels: raster.reverseView.frontInterfacePixels,
          backInterfacePixels: raster.reverseView.backInterfacePixels,
          pairedInterfacePixels: raster.reverseView.pairedInterfacePixels,
          backOnlyInterfacePixels: raster.reverseView.backOnlyInterfacePixels,
          narrowVerticalSlits: raster.reverseView.narrowVerticalSlits,
          enclosedSurfaceHoles: raster.reverseView.enclosedSurfaceHoles,
        } : undefined,
        vertexCount: raster.vertexCount, activeCubeCount: raster.activeCubeCount,
        authorityTransition: raster.globalFineAuthorityTransition,
      } : undefined,
      failures: Object.freeze(failures) });
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "losasso-cutover-oracle", ...losassoCutoverAuthority }));
    if (failures.length > 0) {
      throw new Error(`Losasso cutover oracle rejected: ${JSON.stringify(losassoCutoverAuthority)}`);
    }
    if (!tripwiresDisabled) {
      console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
        phase: "tripwires", backend: "losasso", oracle: "losasso-cutover",
        capturedSteps: steps, required: tripwiresForcedRequired,
        tripped: 0, mode: "native-terminal-receipt" }));
    }
  }
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
    const tripped: TripwireTrip[] = [];
    if (tripwireSteps.length === 0) {
      tripwireSnapshot.destroy();
      tripwireLiveReadback?.destroy();
      if (tripwiresRequired) {
        throw new Error("tripwires could not be evaluated: no accepted step captured a"
          + " tripwire record (see docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3)");
      }
    } else {
      // Reaching here means every captured step already passed its own live
      // evaluation, so nothing fatal can be found now. The ring walk remains
      // the whole-run report: it is what enumerates allow-listed trips across
      // the complete window, from the same records and the same decoder, and
      // it is the forensics source if the live path is ever made conditional.
      const snapshotBytes = tripwireSteps.length * TRIPWIRE_RECORD.strideBytes;
      try {
        await tripwireSnapshot.mapAsync(GPUMapMode.READ, 0, snapshotBytes);
        const mapped = new Uint8Array(tripwireSnapshot.getMappedRange(0, snapshotBytes));
        const words = (record: number, offsetBytes: number, byteLength: number) => new Uint32Array(
          mapped.buffer, mapped.byteOffset + record * TRIPWIRE_RECORD.strideBytes + offsetBytes,
          byteLength / 4,
        );
        for (let record = 0; record < tripwireSteps.length; record += 1) {
          // Exactly one walk emits FLUID_FINE_TOPOLOGY_TRACE, and it is whichever
          // one runs. Under fail-fast the live path already emitted this record
          // at its own step -- which is why the trace survives a fatal step --
          // so the ring must stay silent or it would duplicate every line. With
          // fail-fast off the live path never ran, and the ring is the only
          // walk there is; emitting nothing would silently drop the trace.
          tripped.push(...evaluateTripwireRecord(
            (offsetBytes, byteLength) => words(record, offsetBytes, byteLength),
            tripwireSteps[record]!, tripwireFineGenerations[record]!,
            !tripwiresFailFast));
        }
      } finally {
        if (tripwireSnapshot.mapState === "mapped") tripwireSnapshot.unmap();
        tripwireSnapshot.destroy();
        tripwireLiveReadback?.destroy();
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
      // Under `FLUID_TRIPWIRES=1` this is the gate: the run completed, and the
      // ring says it should not have. Under `failfast` it is defence in depth --
      // the live path already failed at the step that tripped, so a trip
      // reaching here means the two walks disagreed about identical bytes.
      // Either way it is a failed run, never a warning.
      throw tripwireFailure(failing, `over ${tripwireSteps.length} captured steps`);
    }
    console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
      phase: "tripwires", capturedSteps: tripwireSteps.length,
      required: tripwiresRequired, allowed: Array.from(tripwireAllowList),
      tripped: tripped.length,
      // Which mode produced this run's wall. A throughput number is only
      // comparable to another taken in the same mode: fail-fast fences every
      // step and measured +26.8% on the large lane.
      mode: tripwiresFailFast ? "failfast" : "end-of-run",
      // The measured price of fail-fast: the per-step queue fence (solver work
      // waited for early, left in simulationWall_ms) and the host decode
      // (diagnostics, subtracted from it). Both are zero in end-of-run mode.
      liveFence_ms: Math.round(tripwireLiveFence_ms),
      liveDecode_ms: Math.round(tripwireLiveDecode_ms) }));
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
  let finalPerformanceAuthority: Readonly<Record<string, unknown>> | undefined;
  if (performanceProfileRequested && method.id === "octree") {
    if (losassoCutoverLane) {
      finalPerformanceAuthority = losassoCutoverAuthority;
      console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
        phase: "final-performance-authority", ...finalPerformanceAuthority }));
    } else {
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
        readPowerFrontierFailure(): Promise<{
          rowDelta: number[]; finePageDelta: number[]; structuredDispatch: number[];
          descriptorCandidate: number[]; topologyCandidate: number[];
          structuredCandidate: number[]; boundaryCandidate: number[];
          spgridCandidate: number[]; coarseDelta: number[];
          fineSummaryWorkState: number[]; descriptorStatuses: number[];
          candidateSchedules: number[];
        }>;
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
      if (process.env.FLUID_WORKSET_CENSUS === "1") {
        const snapshot = await projection.readPowerFrontierFailure();
        console.log(JSON.stringify({ scenario: scenarioId, method: method.id,
          phase: "settled-maintenance-census", rowDelta: snapshot.rowDelta,
          finePageDelta: snapshot.finePageDelta,
          structuredDispatch: snapshot.structuredDispatch,
          descriptorCandidate: snapshot.descriptorCandidate,
          topologyCandidate: snapshot.topologyCandidate,
          structuredCandidate: snapshot.structuredCandidate,
          boundaryCandidate: snapshot.boundaryCandidate,
          spgridCandidate: snapshot.spgridCandidate,
          coarseDelta: snapshot.coarseDelta,
          fineSummaryWorkState: snapshot.fineSummaryWorkState,
          descriptorStatuses: snapshot.descriptorStatuses,
          candidateSchedules: snapshot.candidateSchedules }));
      }
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
  // Neither compact octree backend dispatches the dense velocity reduction.
  // Exact QA checkpoints already reconstruct the accepted compact rows on the
  // fine lattice, so reuse that backend-neutral evidence rather than adding a
  // production pass, fallback, or second readback solely for scalar telemetry.
  if (method.id === "octree") {
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
  // Losasso deliberately has no Power fine-volume receipt. Generic smoke QA
  // already holds an exact matched/final field, so when solver-native volume
  // telemetry is absent publish the same represented-volume quantity from
  // that evidence instead of leaving a backend-dependent hole in the result.
  const representedVolumeEvidence = final?.summary ?? matched.summary;
  const initialVolumeCellSum = info.initialVolumeCellSum ?? Number.NaN;
  if (!Number.isFinite(info.representedVolumeDrift ?? NaN)
    && Number.isFinite(initialVolumeCellSum)
    && Number.isFinite(representedVolumeEvidence.cellSum)) {
    const reference = Math.max(1, Math.abs(initialVolumeCellSum));
    info.representedVolumeCellSum = representedVolumeEvidence.cellSum;
    info.representedVolumeDrift = (representedVolumeEvidence.cellSum - initialVolumeCellSum) / reference;
  }
  let terminalCompactVelocity: Awaited<ReturnType<typeof readCompactOctreeVelocityField3D>>;
  if (method.id === "octree" && final
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
    && final && !terminalCompactVelocity) {
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
    ? losassoCutoverRaster
      ?? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies)
    : undefined;
  // Always captured for octree: the structured-validation gates require the
  // final generation diagnostics, and a gate that reads `undefined` reports
  // a wiring failure rather than evaluating the solver's actual state.
  const finalGlobalFineGeneration = method.id === "octree"
    ? await readGlobalFineGenerationDiagnostics(device, solver) : undefined;
  // Prefer the accepted fine-volume receipt when the backend publishes one.
  // The cubic QA reconstruction and initialVolumeCellSum intentionally use
  // different representations at a cut interface; comparing those two makes
  // the initial representation offset look like transport drift. Reference
  // and current below come from the same native fine-volume generation.
  if (finalGlobalFineGeneration?.volumeInitialized
    && Number.isFinite(finalGlobalFineGeneration.volumeReference ?? NaN)
    && Number.isFinite(finalGlobalFineGeneration.volumeCurrent ?? NaN)) {
    const reference_m3 = finalGlobalFineGeneration.volumeReference!;
    const current_m3 = finalGlobalFineGeneration.volumeCurrent!;
    const baseCellVolume_m3 = scene.container.width_m * scene.container.height_m
      * scene.container.depth_m / (info.nx * info.ny * info.nz);
    info.referenceLiquidVolume_cells = reference_m3 / baseCellVolume_m3;
    info.representedVolumeCellSum = current_m3 / baseCellVolume_m3;
    info.representedVolumeDrift = (current_m3 - reference_m3)
      / Math.max(1e-30, Math.abs(reference_m3));
  }
  const finalGlobalFineRaster = rasterInitialFinalRequested && method.id === "octree"
    ? losassoCutoverRaster
      ?? await smokeRenderHybridPresentation(instrumentedDevice, solver, scene, bodies,
        verifyGlobalFineGenerationTransition)
    : undefined;
  if (process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1" && method.id === "octree") {
    const levelSet = solver.surfaceFieldTexture
      ? await readFloatTexture3D(device, solver.surfaceFieldTexture, info.nx, info.ny, info.nz)
      : undefined;
    if (levelSet) {
      let exactMismatchCount = 0, maximumAbsoluteError = 0;
      for (let z = 0; z < info.nz; z += 1) for (let y = 0; y < info.ny; y += 1) {
        for (let x = 0; x < info.nx; x += 1) {
          const source = levelSet[x + info.nx * (y + info.ny * z)]!;
          for (const [tx, tz] of [[info.nx - 1 - x, z], [x, info.nz - 1 - z], [z, x]] as const) {
            const target = levelSet[tx + info.nx * (y + info.ny * tz)]!;
            if (!Object.is(source, target)) exactMismatchCount += 1;
            maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(source - target));
          }
        }
      }
      console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
        phase: "fluid-symmetry-bootstrap-level-set",
        metrics: { exactMismatchCount, maximumAbsoluteError } }));
    }
    const coarseSource = solver.coarseLevelSetSource;
    if (coarseSource) {
      const directory = coarseSource.directory.buffer;
      const bytes = await readBufferBinding(device, { buffer: directory }, directory.size);
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const entryCount = (words.length - 8) / 8, volume = info.nx * info.ny * info.nz;
      const denseStart = entryCount - volume;
      let exactMismatchCount = 0, maximumAbsoluteError = 0;
      let first: Record<string, unknown> | undefined;
      let worst: Record<string, unknown> | undefined;
      for (let z = 0; z < info.nz; z += 1) for (let y = 0; y < info.ny; y += 1) {
        for (let x = 0; x < info.nx; x += 1) {
          const at = x + info.nx * (y + info.ny * z);
          const source = floats[8 + 8 * (denseStart + at) + 2]!;
          for (const [transform, tx, tz] of [
            ["reflect-x", info.nx - 1 - x, z],
            ["reflect-z", x, info.nz - 1 - z],
            ["swap-xz", z, x],
          ] as const) {
            const targetAt = tx + info.nx * (y + info.ny * tz);
            const target = floats[8 + 8 * (denseStart + targetAt) + 2]!;
            const absoluteError = Math.abs(source - target);
            if (!Object.is(source, target)) {
              exactMismatchCount += 1;
              const detail = { transform, source: [x, y, z], target: [tx, y, tz],
                sourceValue: source, targetValue: target, absoluteError };
              first ??= detail;
              if (!worst || absoluteError > Number(worst.absoluteError)) worst = detail;
            }
            maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
          }
        }
      }
      console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
        phase: "fluid-symmetry-coarse-level-set", generation: words[1],
        metrics: { exactMismatchCount, maximumAbsoluteError, first, worst } }));
    }
    const pair = (solver as GPUSolverInstance & { globalFineSourceDebugPair?: {
      a: WebGPUFineLevelSetBrickSource; b: WebGPUFineLevelSetBrickSource; publishedIsA: boolean;
    } }).globalFineSourceDebugPair;
    if (pair) {
      const transported = pair.publishedIsA ? pair.b : pair.a;
      console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
        phase: "fluid-symmetry-fine-transport",
        metrics: await readFinePhiSymmetrySource(device, transported) }));
    }
  }
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
  const gpuDataFlowManifest = dataFlowAudit?.report(
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
        airSupportFallbacks?: readonly number[];
        airSupportTopologyFailureLatch?: readonly number[];
        airSupportFailureCounts?: readonly number[];
        airSupportFailureTopology?: Record<string, unknown>;
      } | undefined>;
      readSPGridHierarchyCensus(): Promise<{
        levels: readonly Readonly<Record<string, number>>[];
      } | undefined>;
      readSPGridTouchedDirectoryTripwire(): Promise<{
        enabled: boolean; active: boolean; brickHeader: readonly number[]; pageHeader: readonly number[];
        brickControl: readonly number[]; pageControl: readonly number[];
      } | undefined>;
      readPowerHybridCensus(): Promise<{
        regularRows: number; identityRows: number; powerRows: number;
        liquidRows: number; liveRows: number;
        fullDescriptorRows: number; hybridDescriptorRows: number;
        fullCatalogRows: number; hybridCatalogRows: number;
        fullPageSlotChains: number; hybridPageSlotChains: number;
        epoch: number; machineryReduction: number;
      } | null>;
      readPersistentBandCensus(): Promise<{
        bandRows: number; regularBandRows: number;
        coarseRegularBandRows: number; regularShare: number;
      } | null>;
      readPowerHybridClassSymmetry(): Promise<Record<string, unknown> | undefined>;
      readCoarseSurfaceTrackerReceipt(): Promise<Record<string, unknown> | undefined>;
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
  // Asking for this tripwire and getting silence is the failure mode the whole
  // tripwire doctrine exists to kill: the differential builder ran, the run
  // passed every other gate, and nothing compared the two directories. Both
  // links in the chain are optional (`octreeProjection?`, and inside it
  // `firstOrderVCycle?`), so an unevaluable read has to be loud. `=1` is an
  // explicit request for evidence; produce it or fail.
  const touchedDirectoryTripwire = process.env.FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE === "1"
    ? await diagnosticProjection.octreeProjection?.readSPGridTouchedDirectoryTripwire()
    : undefined;
  if (process.env.FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE === "1") {
    if (touchedDirectoryTripwire === undefined) {
      throw new Error("FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE=1 could not be evaluated:"
        + " no octree projection or no first-order V-cycle on this lane."
        + " The touched-directory differential was NOT checked; do not read this run as correctness evidence.");
    }
    if (!touchedDirectoryTripwire.enabled) {
      throw new Error("FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE=1 but the touched-directory path is disabled;"
        + " set FLUID_SPGRID_TOUCHED_RADIX_SORT=1 so there are two builders to compare.");
    }
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "spgrid-touched-directory-tripwire", metrics: touchedDirectoryTripwire }));
  }
  const powerHybridCensus = process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1"
    || process.env.FLUID_POWER_HYBRID_CENSUS === "1"
    || process.env.FLUID_POWER_HYBRID_MIN_REDUCTION !== undefined
    ? await diagnosticProjection.octreeProjection?.readPowerHybridCensus()
    : undefined;
  // Asking for the band census and getting silence is the same trap the
  // touched-directory tripwire above exists to kill, and it had already sprung:
  // `readPersistentBandCensus` had no caller anywhere in `lib/` or `tools/`, so
  // `FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS=census` published four GPU words that
  // nothing ever read. A mode whose whole purpose is to be measured before the
  // arithmetic moves must emit or fail.
  const persistentBandCensusRequested =
    process.env.FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS !== undefined
    && process.env.FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS !== "0";
  const persistentBandCensus = persistentBandCensusRequested
    ? await diagnosticProjection.octreeProjection?.readPersistentBandCensus()
    : undefined;
  if (persistentBandCensusRequested) {
    if (!persistentBandCensus) {
      throw new Error("FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS selected a mode but no band census"
        + " was published: no octree projection, no persistent executor, or the solve failed"
        + " before P4b. The class-0 band share was NOT measured; do not read this run as evidence.");
    }
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "persistent-band-census", metrics: persistentBandCensus }));
  }
  const powerHybridClassSymmetry = process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1"
    ? await diagnosticProjection.octreeProjection?.readPowerHybridClassSymmetry()
    : undefined;
  if (powerHybridCensus) console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
    phase: "power-hybrid-census", metrics: powerHybridCensus }));
  const minimumPowerHybridReductionText = process.env.FLUID_POWER_HYBRID_MIN_REDUCTION;
  if (minimumPowerHybridReductionText !== undefined) {
    const minimumPowerHybridReduction = Number(minimumPowerHybridReductionText);
    const verdict = octreePowerHybridWorkVerdict(powerHybridCensus, minimumPowerHybridReduction);
    if (!verdict.accepted) throw new Error(`Power-hybrid work gate failed: ${verdict.reason}`);
  }
  if (powerHybridClassSymmetry) console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
    phase: "power-hybrid-class-symmetry", metrics: powerHybridClassSymmetry }));
  // Factor-one only. `completions` below `advances` is the signature of a
  // surface the raster held rather than refreshed, which is what an apparent
  // two-state flicker looks like from inside the tracker.
  const coarseTrackerReceipt = process.env.FLUID_COARSE_TRACKER_RECEIPT === "1"
    ? await diagnosticProjection.octreeProjection?.readCoarseSurfaceTrackerReceipt()
    : undefined;
  if (coarseTrackerReceipt) console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
    phase: "coarse-tracker-receipt", metrics: coarseTrackerReceipt }));
  const finePlan = diagnosticProjection.globalFineLevelSetSource?.plan;
  const algorithmDiagnostics = terminalAlgorithmState || spgridHierarchy || powerHybridCensus ? {
    topologyControl: terminalAlgorithmState?.topologyControl,
    structuredVelocityControl: terminalAlgorithmState?.structuredVelocityControl,
    structuredBoundaryControl: terminalAlgorithmState?.structuredBoundaryControl,
    airSupportControl: terminalAlgorithmState?.airSupportControl,
    airSupportTerminalScratch: terminalAlgorithmState?.airSupportTerminalScratch,
    // Both latches were authored with no consumer. They are the only record of
    // why a Section 5 transaction declined, and the transaction declines often
    // enough on the factor-one lane to hold the rendered surface.
    airSupportFallbacks: terminalAlgorithmState?.airSupportFallbacks,
    airSupportTopologyFailureLatch: terminalAlgorithmState?.airSupportTopologyFailureLatch,
    airSupportFailureCounts: terminalAlgorithmState?.airSupportFailureCounts,
    airSupportFailureTopology: terminalAlgorithmState?.airSupportFailureTopology,
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
    powerHybridCensus,
  } : undefined;
  // A second read at the end of the run. A difference against the
  // after-construction record is a solver that grows its arenas while stepping,
  // which is the one memory behaviour a construction-time snapshot cannot see.
  if (residentMemoryCensus) {
    const residentMemory = residentMemoryCensus.snapshot();
    console.log(JSON.stringify({ scenario: scenarioId, method: resultMethod,
      phase: "gpu-memory", when: "after-run", residentMemory }));
    console.log(formatResidentMemoryReport(residentMemory));
  }
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
    gpuPassBoundaryAudit: passBrokerBoundaryAuditSnapshot(),
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
