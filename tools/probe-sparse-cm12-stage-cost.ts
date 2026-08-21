/**
 * Where does a Sparse CM12 advance spend its time, per hardware stage,
 * averaged over many advances and on a scene large enough that non-pressure
 * stages clear Dawn's 65.5 us timestamp tick?
 *
 * `probe-sparse-cm12-stage-trace.ts` is the acceptance gate for the partition
 * existing at all; it samples the mini 32^3 scene and reports one trace. This
 * is the measurement lane: pick a scene, sample every advance, and report the
 * median stage cost with the pressure solve separated from everything else.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-cm12-stage-cost.ts --scene=long-dam
 *
 * Shipping ocean B16/P16 non-pressure gate (24 hardware samples minimum):
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-cm12-stage-cost.ts \
 *     --scene=ocean-seiche --brick-fine=16 --presentation-page=16 \
 *     --warmup=8 --frames=24 --enforce-non-pressure-gate=1
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  fluidPipelinePhaseCosts,
  measureFluidPipelineStage,
} from "../lib/core/fluid-pipeline";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDamBreakBox } from "../lib/core/initial-fluid";
import {
  GPUStageTimestampRecorder,
  type PerformanceTrace,
} from "../lib/core/performance-trace";
import {
  createMinimalPowerDamBreakScene,
  createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene,
  createLargePowerHydrostaticScene,
  createDeepPowerHydrostaticScene,
  createOceanSeicheScene,
  createSparseCM12LongDamBreakScene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  gpuCompilationManagerFor,
  invalidateGPUCompilationManager,
} from "../lib/core/gpu-compilation-manager";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  createProcessRetainedDawnGPU,
  type NodeDawnProvider,
} from "../lib/harness/node-dawn-provider";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import type { SparseCM12TransportExperiment } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { inspectSparseCM12PressureCutoverAuthorities } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-cutover-observability";
import {
  SPARSE_CM12_FRAME_CONTROL_PHASE,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { SPARSE_CM12_SCALAR_RESULT_PHASE } from
  "../lib/methods/adaptive-mass/sparse-cm12-scalar-result-receipts";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(`Sparse CM12 hardware stage-cost probe

Usage:
  WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \\
  node --import tsx tools/probe-sparse-cm12-stage-cost.ts [options]

Options:
  --help, -h                         Print this help and exit without acquiring WebGPU
  --scene=NAME                       mini16, mini32, mini64, long-dam, ocean-seiche,
                                     ocean, or symmetric-expansion (default long-dam)
  --brick-fine=4|8|16                Sparse brick ladder (default 16)
  --presentation-page=4|8|16         Presentation page size (default 16)
  --transport-experiment=NAME        Construction-static A/B specialization
                                     (default baseline)
  --warmup=N                         Warmup hardware samples (default 8)
  --frames=N                         Measured hardware samples (default 40)
  --capture-gap-ms=N                 Timestamp capture spacing (default 110)
  --minimum-cell-size=N              Add a minimum-cell-size region in finest
                                     cells (power of two; omitted by default)
  --region-scope=domain|initial-dam  Region bounds (default domain)
  --max-non-pressure-ms=N            Eligible gate threshold (default 10)
  --enforce-non-pressure-gate=0|1    Exit nonzero when the eligible gate fails
  --enforce-pressure-receipts=0|1    Require fault-free PTR/FPA/PCF/PCA receipts
                                     (default 1)
  --out=PATH                         Write the JSON receipt

The non-pressure gate is eligible only for ocean-seiche B16/P16 with at least
24 hardware-timestamped frames. --help performs no GPU work.`);
  process.exit(0);
}

const sceneName = argument("scene", "long-dam");
const warmup = Number(argument("warmup", "8"));
const sampled = Number(argument("frames", "40"));
const brickFineResolution = Number(argument("brick-fine", "16"));
const presentationPageResolution = Number(argument("presentation-page", "16"));
const transportExperiment = argument("transport-experiment", "baseline") as
  SparseCM12TransportExperiment;
const captureGap_ms = Number(argument("capture-gap-ms", "110"));
const minimumCellSize = Number(argument("minimum-cell-size", "0"));
const regionScope = argument("region-scope", "domain");
const maximumTarget_ms = Number(argument("max-non-pressure-ms",
  argument("max-target-ms", "10")));
const enforceNonPressureGate = argument("enforce-non-pressure-gate",
  argument("enforce-target-gate", "0")) === "1";
const enforcePressureReceipts = argument("enforce-pressure-receipts", "1") === "1";
const outputPath = argument("out", "");
for (const [name, value] of Object.entries({ warmup, sampled, brickFineResolution,
  presentationPageResolution, captureGap_ms, maximumTarget_ms })) {
  if (!(Number.isFinite(value) && value > 0)) throw new RangeError(
    `${name} must be finite and positive; received ${value}`,
  );
}
if (!(Number.isSafeInteger(minimumCellSize) && minimumCellSize >= 0
  && (minimumCellSize === 0 || (minimumCellSize & (minimumCellSize - 1)) === 0))) {
  throw new RangeError("minimum-cell-size must be zero or a positive power of two");
}
if (regionScope !== "domain" && regionScope !== "initial-dam") {
  throw new RangeError("region-scope must be domain or initial-dam");
}
if (![4, 8, 16].includes(brickFineResolution)
  || ![4, 8, 16].includes(presentationPageResolution)
  || presentationPageResolution > brickFineResolution
  || brickFineResolution % presentationPageResolution !== 0) {
  throw new RangeError("brick-fine and presentation-page must be compatible values in 4, 8, 16");
}
if (!["baseline", "mass-swept-clean", "structure-sharpening-legacy",
  "face-row-packets", "face-direct-preparation", "face-characteristic-cache",
  "structure-face-cache-legacy",
  "structure-mass-swept-legacy", "activity-scalar-bricks",
  "structure-activity-scalar-legacy", "presentation-uniform-bulk",
  "structure-presentation-uniform-legacy"]
  .concat(["mass-rung-packets", "mass-rung-local"])
  .includes(transportExperiment)) {
  throw new RangeError(
    "unsupported stage-cost transport experiment",
  );
}
const buildScene = sceneName === "mini16" ? createMinimalPowerDamBreakScene
  : sceneName === "mini32" ? createMinimalPowerDamBreak32Scene
  : sceneName === "mini64" ? createMinimalPowerDamBreak64Scene
  : sceneName === "long-dam" ? createSparseCM12LongDamBreakScene
  : sceneName === "large-hydrostatic" ? createLargePowerHydrostaticScene
  : sceneName === "deep-hydrostatic" ? createDeepPowerHydrostaticScene
  : sceneName === "ocean" || sceneName === "ocean-seiche" ? createOceanSeicheScene
  : sceneName === "symmetric-expansion" ? createSymmetricExpansionScene
  : undefined;
if (!buildScene) throw new RangeError(
  `scene must be mini16, mini32, mini64, long-dam, large-hydrostatic, `
    + `deep-hydrostatic, ocean-seiche, ocean, or symmetric-expansion; received ${sceneName}`,
);

const median = (values: number[]): number => {
  if (values.length === 0) return Number.NaN;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor((ordered.length - 1) / 2)]!;
};

const percentile = (values: number[], quantile: number): number => {
  if (values.length === 0) return Number.NaN;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(quantile * ordered.length) - 1]!;
};

const TARGET_STAGE_IDS = Object.freeze([
  "receiver-topology",
  "scalar-transport",
  "gamma-diffusion",
  "surface-conditioning",
  "pressure-topology",
  "presentation-publication",
] as const);
const PRESSURE_TOPOLOGY_PHASE_LABEL = "Composite pressure topology + ghost-fluid rows";
const REPOSITORY_ROOT = fileURLToPath(new URL("..", import.meta.url));
const PINNED_BASELINE_PATH =
  "artifacts/sparse-cm12-ocean-b16-p16-stage-cost-baseline.json";
const PINNED_BASELINE_SHA256 =
  "87f2463d688dd36c2313258550992eaab8a9e9c8581888ba553c4f254ebbf4e2";
type FrameControlHeader = {
  readonly phase: number; readonly fault: number; readonly firstFaultOwner: number;
  readonly acceptedGeneration: number; readonly candidateGeneration: number;
  readonly sealedGeneration: number; readonly scalarParity: number;
  readonly faceParity: number; readonly coverage: number; readonly committedFrames: number;
};
type ScalarAuthorityHeader = {
  readonly phase: number; readonly fault: number;
  readonly firstFaultTile: number;
  readonly acceptedGeneration: number; readonly candidateGeneration: number;
  readonly topologyGeneration: number; readonly sourceParity: number;
};
type StageCostQASolver = {
  readSparseWorkShapeQA(): {
    readonly finestDomainCellCount: number;
    readonly logicalBrickDimensions: readonly number[];
    readonly logicalBrickCount: number;
    readonly packedBrickCount: number;
    readonly templateCellCount: number;
    readonly templateRowCount: number;
    readonly templateCellWorkgroups: number;
    readonly templateRowWorkgroups: number;
    readonly conditioningClearBytesPerFrame: number;
    readonly pressureScratchClearBytesPerFrame: number;
    readonly pressureHierarchyGroupCount: number;
    readonly pressureHierarchyEdgeCount: number;
    readonly pressureFineEdgeCount: number;
    readonly pressureCoarseEdgeCount: number;
    readonly scalarResultTileCapacity: number;
    readonly facePreparationLeafCount: number;
    readonly presentationPageCount: number;
    readonly allocatedBytes: number;
  };
  readAdaptiveRepresentationQA(): Promise<Record<string, number | boolean>>;
  readFrameControlQA(): Promise<FrameControlHeader>;
  readScalarAuthorityHeaderQA(): Promise<ScalarAuthorityHeader>;
  readSharpeningCellAuthorityQA(): Promise<{
    readonly generation: number; readonly packetCount: number;
  }>;
  readScalarIngressHeaderQA(): Promise<{
    readonly candidateGeneration: number; readonly eventCount: number;
    readonly fault: number; readonly firstFaultTile: number;
  }>;
  readScalarIngressEventsQA(): Promise<readonly {
    readonly tile: number; readonly generation: number; readonly causeMask: number;
  }[]>;
  readVelocityExtensionHeaderQA(): Promise<{
    readonly flags: number; readonly phase: number;
    readonly acceptedGeneration: number; readonly candidateGeneration: number;
    readonly topologyGeneration: number; readonly capacity: number;
    readonly rootCount: number; readonly blastCount: number; readonly maximumDepth: number;
    readonly executedCellCount: number; readonly reusedCellCount: number;
    readonly faultCount: number; readonly uncoveredWriteCount: number;
    readonly firstFault?: { readonly cell: number; readonly depth: number };
    readonly framePlanProvenance?: Record<string, number>;
  }>;
  readVelocityExtensionQA(): Promise<{
    readonly header: Uint32Array;
    readonly blastDepth: Uint32Array; readonly candidateDepth: Uint32Array;
    readonly rootCause: Uint32Array; readonly acceptedDepth: Uint32Array;
    readonly acceptedOwner: Uint32Array; readonly velocityBits: Uint32Array;
  }>;
};
const gitCommit = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: REPOSITORY_ROOT, encoding: "utf8",
}).trim();
const gitStatus = execFileSync("git", ["status", "--porcelain", "--untracked-files=all"], {
  cwd: REPOSITORY_ROOT, encoding: "utf8",
});
const pinnedBaselineSha256 = createHash("sha256").update(await readFile(
  new URL(`../${PINNED_BASELINE_PATH}`, import.meta.url),
)).digest("hex");
assert.equal(pinnedBaselineSha256, PINNED_BASELINE_SHA256,
  `pinned 08:05 baseline ${PINNED_BASELINE_PATH} changed`);

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-sparse-cm12-stage-cost.ts");
let device: GPUDevice | undefined;
let teardownSolver: { destroy(): void } | undefined;
try {
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(
    dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`],
  );
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  assert.ok(adapter.features.has("timestamp-query"),
    "stage-cost measurement requires hardware timestamp queries; queue-wall fallback is not accepted");
  device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query" as GPUFeatureName],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });
  await GPUStageTimestampRecorder.prepare(device);

  const scene = buildScene();
  if (minimumCellSize > 0) {
    const containerMinimum = {
      x: -0.5 * scene.container.width_m,
      y: 0,
      z: -0.5 * scene.container.depth_m,
    };
    const dam = sceneDamBreakBox(scene);
    const min_m = regionScope === "domain" ? containerMinimum : {
      x: containerMinimum.x + dam.min.x * scene.container.width_m,
      y: dam.min.y * scene.container.height_m,
      z: containerMinimum.z + dam.min.z * scene.container.depth_m,
    };
    const max_m = regionScope === "domain" ? {
      x: 0.5 * scene.container.width_m,
      y: scene.container.height_m,
      z: 0.5 * scene.container.depth_m,
    } : {
      x: containerMinimum.x + dam.max.x * scene.container.width_m,
      y: dam.max.y * scene.container.height_m,
      z: containerMinimum.z + dam.max.z * scene.container.depth_m,
    };
    scene.fluid.refinementRegions = [{
      id: `stage-cost-min-${minimumCellSize}-${regionScope}`,
      rule: "minimum-cell-size",
      minimumCellSize_cells: minimumCellSize,
      min_m,
      max_m,
    }];
  }
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene",
    brickFineResolution: String(brickFineResolution),
    presentationPageResolution: String(presentationPageResolution),
  });
  const solver = transportExperiment === "baseline"
    ? await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {})
    : await WebGPUAdaptiveMassSolver.createTransportExperimentForQA(
      transportExperiment, device, scene, "balanced", undefined,
      adaptiveMassSolverOptions(values), () => {});
  teardownSolver = solver;
  const qaSolver = solver as typeof solver & StageCostQASolver;
  const workShape = qaSolver.readSparseWorkShapeQA();
  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;

  const stageSamples = new Map<string, number[]>();
  const totals: number[] = [];
  const wallFrameSamples: number[] = [];
  const committedSamples: number[] = [];
  const pressureTopologyInputChangedSamples: boolean[] = [];
  const pressureTopologyWorkSamples: {
    readonly inputAttributionStatus: "matched" | "unavailable";
    readonly inputTopologyGeneration?: number;
    readonly inputTopologyChanged: boolean;
    readonly priorCommittedBricks?: number;
    readonly endFrameTopologyGeneration: number;
    readonly endFrameCommittedBricks: number;
    readonly acceptedCells: number;
    readonly acceptedRows: number;
    readonly temporalCells: number;
    readonly temporalRows: number;
    readonly pcmCellDirtyLeaves: number;
    readonly pcmRowDirtyLeaves: number;
    readonly pressureCells: number;
    readonly pressureRows: number;
    readonly ptrPhase: number;
    readonly ptrFault: number;
    readonly ptrChangedBricks: number;
    readonly ptrChangedRows: number;
    readonly ptrCellExecutions: number;
    readonly ptrRowExecutions: number;
    readonly ptrBrickDirtyLeaves: number;
    readonly ptrRowDirtyLeaves: number;
    readonly pressureAuthorityReceiptComplete: boolean;
    readonly pressureAuthorityReceiptIssues: readonly string[];
    readonly pressureAuthorities?: NonNullable<NonNullable<
      Awaited<ReturnType<typeof solver.readStats>>["adaptivePressureTopologyAttribution"]
    >["authorities"]>;
  }[] = [];
  const phaseSamples = new Map<string, number[]>();
  const cpuPhaseSamples = new Map<string, number[]>();
  const cpuStageSamples = new Map<string, number[]>();
  const targetSamples: number[] = [];
  const cpuTargetSamples: number[] = [];
  const nonPressureSamples: number[] = [];
  const cpuNonPressureSamples: number[] = [];
  const closureErrors: number[] = [];
  const cpuBySampleId = new Map<number, PerformanceTrace>();
  const authoritySamples: Array<{
    readonly advance: number; readonly expectedFrameControlGeneration: number;
    readonly expectedScalarAuthorityGeneration: number;
    readonly frameControl: FrameControlHeader & { readonly stalled: boolean;
      readonly successorMatched: boolean; readonly valid: boolean };
    readonly scalarAuthority: ScalarAuthorityHeader & { readonly stalled: boolean;
      readonly successorMatched: boolean; readonly valid: boolean };
    readonly scalarIngressEventCount: number;
  }> = [];
  let firstAuthorityFailure: Record<string, unknown> | undefined;
  let diagnosticFailure: string | undefined;
  let lastScalarAuthorityHeader: ScalarAuthorityHeader | undefined;
  let seen = 0;
  let priorFrameCommitted = 0;
  let priorTraceSampleId = 0;
  let finalInfo: Awaited<ReturnType<typeof solver.readStats>> | undefined;
  // Seed the readback-only topology attribution with construction step 0.
  // This encodes no physics and makes a no-warmup first sample attributable.
  await solver.readStats();
  const [initialFrameControl, initialScalarAuthority] = await Promise.all([
    qaSolver.readFrameControlQA(), qaSolver.readScalarAuthorityHeaderQA(),
  ]);
  let priorFrameControlGeneration = initialFrameControl.acceptedGeneration;
  let priorScalarAuthorityGeneration = initialScalarAuthority.acceptedGeneration;
  lastScalarAuthorityHeader = {
    phase: initialScalarAuthority.phase, fault: initialScalarAuthority.fault,
    firstFaultTile: initialScalarAuthority.firstFaultTile,
    acceptedGeneration: initialScalarAuthority.acceptedGeneration,
    candidateGeneration: initialScalarAuthority.candidateGeneration,
    topologyGeneration: initialScalarAuthority.topologyGeneration,
    sourceParity: initialScalarAuthority.sourceParity,
  };
  const maximumAdvances = warmup + sampled + 4;
  const debugProgress = process.env.FLUID_STAGE_PROBE_DEBUG === "1";
  const debug = (message: string) => {
    if (debugProgress) process.stderr.write(`[stage-probe] ${message}\n`);
  };
  for (let frame = 1; frame <= maximumAdvances; frame += 1) {
    const wallFrameStarted_ms = performance.now();
    debug(`advance ${frame} begin`);
    while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
    debug(`advance ${frame} encoded`);
    await device.queue.onSubmittedWorkDone();
    if (frame > warmup && frame <= warmup + sampled) {
      wallFrameSamples.push(performance.now() - wallFrameStarted_ms);
    }
    debug(`advance ${frame} queue complete`);
    const [frameControl, scalarAuthorityQA, scalarIngress] = await Promise.all([
      qaSolver.readFrameControlQA(), qaSolver.readScalarAuthorityHeaderQA(),
      qaSolver.readScalarIngressHeaderQA(),
    ]);
    debug(`advance ${frame} authority headers complete`);
    const expectedFrameControlGeneration = initialFrameControl.acceptedGeneration + frame;
    // SRR1 has no accepted construction result (generation 0); its first
    // physics commit therefore bootstraps directly to FCA generation 2.
    // Thereafter it advances by exactly one with FCA.
    const expectedScalarAuthorityGeneration = expectedFrameControlGeneration;
    const frameControlStalled = frameControl.acceptedGeneration <= priorFrameControlGeneration;
    const scalarAuthorityStalled = scalarAuthorityQA.acceptedGeneration
      <= priorScalarAuthorityGeneration;
    const frameControlSuccessorMatched = frameControl.acceptedGeneration
        === expectedFrameControlGeneration
      && frameControl.candidateGeneration === expectedFrameControlGeneration
      && frameControl.sealedGeneration === expectedFrameControlGeneration
      && frameControl.committedFrames === initialFrameControl.committedFrames + frame;
    const scalarAuthoritySuccessorMatched = scalarAuthorityQA.acceptedGeneration
        === expectedScalarAuthorityGeneration
      && scalarAuthorityQA.candidateGeneration === expectedScalarAuthorityGeneration
      && (scalarAuthorityQA.acceptedGeneration === priorScalarAuthorityGeneration + 1
        || (frame === 1 && priorScalarAuthorityGeneration === 0
          && scalarAuthorityQA.acceptedGeneration === expectedFrameControlGeneration));
    const frameControlValid = frameControl.phase === SPARSE_CM12_FRAME_CONTROL_PHASE.accepted
      && frameControl.fault === 0
      && frameControl.firstFaultOwner === 0xffff_ffff
      && !frameControlStalled && frameControlSuccessorMatched;
    const scalarAuthorityValid = scalarAuthorityQA.phase === SPARSE_CM12_SCALAR_RESULT_PHASE.accepted
      && scalarAuthorityQA.fault === 0 && !scalarAuthorityStalled
      && scalarAuthoritySuccessorMatched
      && scalarAuthorityQA.acceptedGeneration === frameControl.acceptedGeneration;
    const scalarAuthorityHeader = {
      phase: scalarAuthorityQA.phase, fault: scalarAuthorityQA.fault,
      firstFaultTile: scalarAuthorityQA.firstFaultTile,
      acceptedGeneration: scalarAuthorityQA.acceptedGeneration,
      candidateGeneration: scalarAuthorityQA.candidateGeneration,
      topologyGeneration: scalarAuthorityQA.topologyGeneration,
      sourceParity: scalarAuthorityQA.sourceParity,
    };
    lastScalarAuthorityHeader = scalarAuthorityHeader;
    authoritySamples.push({
      advance: frame, expectedFrameControlGeneration, expectedScalarAuthorityGeneration,
      frameControl: { ...frameControl, stalled: frameControlStalled,
        successorMatched: frameControlSuccessorMatched, valid: frameControlValid },
      scalarAuthority: { ...scalarAuthorityHeader, stalled: scalarAuthorityStalled,
        successorMatched: scalarAuthoritySuccessorMatched, valid: scalarAuthorityValid },
      scalarIngressEventCount: scalarIngress.eventCount,
    });
    priorFrameControlGeneration = frameControl.acceptedGeneration;
    priorScalarAuthorityGeneration = scalarAuthorityQA.acceptedGeneration;
    if ((!frameControlValid || !scalarAuthorityValid) && firstAuthorityFailure === undefined) {
      try {
        const vexHeader = await qaSolver.readVelocityExtensionHeaderQA();
        let firstFaultDetail: Record<string, unknown> | undefined;
        if (vexHeader.firstFault) {
          // The capacity-sized comparison payload is read at most once, and
          // only when VEX1 itself supplied an exact first-fault cell.
          const vex = await qaSolver.readVelocityExtensionQA();
          const cell = vexHeader.firstFault.cell;
          firstFaultDetail = {
            cell, depth: vexHeader.firstFault.depth,
            blastDepth: vex.blastDepth[cell], candidateDepth: vex.candidateDepth[cell],
            rootCause: vex.rootCause[cell], acceptedDepth: vex.acceptedDepth[cell],
            acceptedOwner: vex.acceptedOwner[cell],
            acceptedVelocityBits: [...vex.velocityBits.slice(4 * cell, 4 * cell + 4)],
          };
        }
        firstAuthorityFailure = {
          advance: frame,
          frameControl: authoritySamples.at(-1)!.frameControl,
          scalarAuthority: authoritySamples.at(-1)!.scalarAuthority,
          velocityExtension: {
            header: vexHeader,
            firstFault: firstFaultDetail,
          },
        };
      } catch (error) {
        firstAuthorityFailure = {
          advance: frame,
          frameControl: authoritySamples.at(-1)!.frameControl,
          scalarAuthority: authoritySamples.at(-1)!.scalarAuthority,
          velocityExtensionCaptureError: error instanceof Error ? error.message : String(error),
        };
      }
      diagnosticFailure = `advance ${frame} FCA1/SRR1 successor fault or stall`;
      break;
    }
    if (validationErrors.length > 0) {
      diagnosticFailure = `advance ${frame} emitted WebGPU validation errors`;
      break;
    }
    const pollStarted_ms = performance.now();
    const expectedTraceContext = `adaptive-mass:sim-${(frame * dt_s).toFixed(6)}`;
    let trace: PerformanceTrace | undefined;
    do {
      const candidate = solver.readPerformanceTraceSnapshot!().physicsTrace;
      if (candidate?.measurementSource === "gpu-hardware-timestamp"
        && candidate.sampleId > priorTraceSampleId
        && candidate.context === expectedTraceContext) {
        trace = candidate;
        break;
      }
      const remaining_ms = captureGap_ms - (performance.now() - pollStarted_ms);
      if (remaining_ms <= 0) break;
      await new Promise((resolve) => setTimeout(resolve, Math.min(5, remaining_ms)));
    } while (true);
    debug(`advance ${frame} stats read begin`);
    const info = await solver.readStats();
    debug(`advance ${frame} stats read complete`);
    finalInfo = info;
    const committedThisFrame = (info as {
      adaptiveTopologyCommittedBrickCount?: number }).adaptiveTopologyCommittedBrickCount ?? 0;
    const pressureTopologyAttribution = info.adaptivePressureTopologyAttribution;
    // Topology commits occur after pressure-topology. Its dynamic repair is
    // therefore caused by the prior frame's commit, not the commit reported
    // by this frame's terminal diagnostics.
    const pressureTopologyInputChanged = pressureTopologyAttribution?.status === "matched"
      ? (pressureTopologyAttribution.priorCommittedBrickCount ?? 0) > 0
      : priorFrameCommitted > 0;
    priorFrameCommitted = committedThisFrame;
    if (info.physicsCPUTrace) cpuBySampleId.set(info.physicsCPUTrace.sampleId,
      info.physicsCPUTrace);
    const waitForNextCapture = async () => {
      const remaining_ms = captureGap_ms - (performance.now() - pollStarted_ms);
      if (remaining_ms > 0) {
        await new Promise((resolve) => setTimeout(resolve, remaining_ms));
      }
    };
    if (!trace) {
      await waitForNextCapture();
      continue;
    }
    priorTraceSampleId = trace.sampleId;
    if (trace.sampleId <= warmup) {
      await waitForNextCapture();
      continue;
    }
    const committed = committedThisFrame;
    const costs = fluidPipelinePhaseCosts(trace);
    const stageDurations = new Map<string, number>();
    for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
      const measurement = measureFluidPipelineStage(
        stage, ADAPTIVE_MASS_FLUID_PIPELINE.stages, costs, trace.total_ms, "on");
      stageDurations.set(stage.id, measurement.duration_ms ?? 0);
      const changed = stage.id === "pressure-topology"
        ? pressureTopologyInputChanged : committed > 0;
      const key = changed ? `${stage.id}|changed` : `${stage.id}|quiescent`;
      for (const name of [stage.id, key]) {
        const bucket = stageSamples.get(name) ?? [];
        bucket.push(measurement.duration_ms ?? 0);
        stageSamples.set(name, bucket);
      }
    }
    targetSamples.push(TARGET_STAGE_IDS.reduce((sum, id) =>
      sum + (stageDurations.get(id) ?? 0), 0));
    for (const phase of trace.phases) {
      const changed = phase.label === PRESSURE_TOPOLOGY_PHASE_LABEL
        ? pressureTopologyInputChanged : committed > 0;
      const key = changed ? `${phase.label}|changed` : `${phase.label}|quiescent`;
      for (const name of [phase.label, key]) {
        const bucket = phaseSamples.get(name) ?? [];
        bucket.push(phase.duration_ms);
        phaseSamples.set(name, bucket);
      }
    }
    const accounted_ms = trace.phases.reduce((sum, phase) => sum + phase.duration_ms, 0);
    nonPressureSamples.push(trace.phases.reduce((sum, phase) =>
      sum + (phase.label === "One-reduction sparse MGPCG pressure solve"
        ? 0 : phase.duration_ms), 0));
    closureErrors.push(trace.total_ms - accounted_ms);
    const cpuTrace = cpuBySampleId.get(trace.sampleId);
    if (cpuTrace) {
      const cpuCosts = fluidPipelinePhaseCosts(cpuTrace);
      const cpuDurations = new Map<string, number>();
      for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
        const measurement = measureFluidPipelineStage(
          stage, ADAPTIVE_MASS_FLUID_PIPELINE.stages, cpuCosts, cpuTrace.total_ms, "on");
        cpuDurations.set(stage.id, measurement.duration_ms ?? 0);
        const duration = measurement.duration_ms ?? 0;
        const changed = stage.id === "pressure-topology"
          ? pressureTopologyInputChanged : committed > 0;
        const changedKey = `${stage.id}|${changed ? "changed" : "quiescent"}`;
        for (const name of [stage.id, changedKey]) {
          const bucket = cpuStageSamples.get(name) ?? [];
          bucket.push(duration);
          cpuStageSamples.set(name, bucket);
        }
      }
      cpuTargetSamples.push(TARGET_STAGE_IDS.reduce((sum, id) =>
        sum + (cpuDurations.get(id) ?? 0), 0));
      cpuNonPressureSamples.push(cpuTrace.phases.reduce((sum, phase) =>
        sum + (phase.label === "One-reduction sparse MGPCG pressure solve"
          ? 0 : phase.duration_ms), 0));
      for (const phase of cpuTrace.phases) {
        const changed = phase.label === PRESSURE_TOPOLOGY_PHASE_LABEL
          ? pressureTopologyInputChanged : committed > 0;
        const changedKey = `${phase.label}|${changed ? "changed" : "quiescent"}`;
        for (const name of [phase.label, changedKey]) {
          const bucket = cpuPhaseSamples.get(name) ?? [];
          bucket.push(phase.duration_ms);
          cpuPhaseSamples.set(name, bucket);
        }
      }
    }
    totals.push(trace.total_ms);
    committedSamples.push(committed);
    pressureTopologyInputChangedSamples.push(pressureTopologyInputChanged);
    const pcm = info.adaptivePressureCanonicalMembership;
    const ptr = info.adaptivePressureTopologyRepair;
    const pressureAuthorities = pressureTopologyAttribution?.authorities;
    const pressureAuthorityInspection = inspectSparseCM12PressureCutoverAuthorities(
      pressureAuthorities, pressureTopologyAttribution?.inputTopologyGeneration,
    );
    const pressureReceiptIssues = [...pressureAuthorityInspection.issues];
    if (!ptr) pressureReceiptIssues.push("PTR receipt is unavailable");
    else if (ptr.fault !== 0) pressureReceiptIssues.push(
      `PTR fault ${ptr.fault} at family ${ptr.firstFaultFamily}/id ${ptr.firstFaultId}`,
    );
    const pressureReceiptComplete = pressureAuthorityInspection.complete
      && ptr !== undefined && ptr.fault === 0;
    pressureTopologyWorkSamples.push({
      inputAttributionStatus: pressureTopologyAttribution?.status ?? "unavailable",
      ...(pressureTopologyAttribution?.inputTopologyGeneration === undefined ? {} : {
        inputTopologyGeneration: pressureTopologyAttribution.inputTopologyGeneration,
      }),
      inputTopologyChanged: pressureTopologyInputChanged,
      ...(pressureTopologyAttribution?.priorCommittedBrickCount === undefined ? {} : {
        priorCommittedBricks: pressureTopologyAttribution.priorCommittedBrickCount,
      }),
      endFrameTopologyGeneration: info.adaptiveTopologyShadowGeneration ?? 0,
      endFrameCommittedBricks: committed,
      acceptedCells: info.adaptiveAcceptedCellCount ?? 0,
      acceptedRows: info.adaptiveAcceptedRowCount ?? 0,
      temporalCells: info.adaptiveTemporalScalarCellCount ?? 0,
      temporalRows: info.adaptiveTemporalScalarRowCount ?? 0,
      pcmCellDirtyLeaves: pcm?.cell.dirtyCount ?? 0,
      pcmRowDirtyLeaves: pcm?.row.dirtyCount ?? 0,
      pressureCells: info.adaptivePressureCellCount ?? 0,
      pressureRows: info.adaptivePressureActiveRowCount ?? 0,
      ptrPhase: ptr?.phase ?? 0,
      ptrFault: ptr?.fault ?? 0,
      ptrChangedBricks: ptr?.changedBrickCount ?? 0,
      ptrChangedRows: ptr?.changedRowCount ?? 0,
      ptrCellExecutions: ptr?.cellExecutionCount ?? 0,
      ptrRowExecutions: ptr?.rowExecutionCount ?? 0,
      ptrBrickDirtyLeaves: ptr?.brickDirtyLeafCount ?? 0,
      ptrRowDirtyLeaves: ptr?.rowDirtyLeafCount ?? 0,
      pressureAuthorityReceiptComplete: pressureReceiptComplete,
      pressureAuthorityReceiptIssues: Object.freeze(pressureReceiptIssues),
      ...(pressureAuthorities === undefined ? {} : { pressureAuthorities }),
    });
    seen += 1;
    if (enforcePressureReceipts && !pressureReceiptComplete) {
      diagnosticFailure = `sample ${seen} pressure receipt fault: ${
        pressureReceiptIssues.join("; ")}`;
      break;
    }
    if (seen >= sampled) break;
    await waitForNextCapture();
  }

  if (!diagnosticFailure && seen !== sampled) {
    diagnosticFailure = `captured ${seen}/${sampled} requested hardware traces in ${
      maximumAdvances} advances`;
  }
  const adaptiveRepresentation = await qaSolver.readAdaptiveRepresentationQA();
  const scalarIngressEvents = await qaSolver.readScalarIngressEventsQA();
  const tilesPerAxis = brickFineResolution / 4;
  const tilesPerBrick = tilesPerAxis ** 3;
  const [logicalX, logicalY] = workShape.logicalBrickDimensions;
  const scalarEventYHistogram = Array.from({ length: logicalY! * tilesPerAxis }, () => 0);
  for (const event of scalarIngressEvents) {
    const logical = Math.floor(event.tile / tilesPerBrick);
    const local = event.tile % tilesPerBrick;
    const by = Math.floor(logical / logicalX!) % logicalY!;
    const ty = Math.floor(local / tilesPerAxis) % tilesPerAxis;
    scalarEventYHistogram[by * tilesPerAxis + ty]! += 1;
  }
  if (!diagnosticFailure && cpuTargetSamples.length !== sampled) {
    diagnosticFailure = `matched ${cpuTargetSamples.length}/${sampled} CPU traces by sample id`;
  }
  if (!diagnosticFailure
    && !closureErrors.every((error) => Math.abs(error) < 1e-6)) {
    diagnosticFailure = `hardware stage partition did not close exactly; errors=${
      closureErrors.join(",")}`;
  }

  const stages = [...stageSamples].map(([id, samples]) => ({
    stage: id, median_ms: Number(median(samples).toFixed(4)),
  })).sort((left, right) => right.median_ms - left.median_ms);
  const total = median(totals);
  const pressure = stages.find((stage) => stage.stage === "pressure-solve")?.median_ms ?? 0;
  const nonPressureMedian_ms = median(nonPressureSamples);
  const nonPressureP95_ms = percentile(nonPressureSamples, 0.95);
  const targetMedian_ms = median(targetSamples);
  const targetP95_ms = percentile(targetSamples, 0.95);
  const frameAuthorityStage = stages.find(
    (stage) => stage.stage === "transport-velocity-extension");
  const frameAuthorityCPU = [...cpuPhaseSamples].find(
    ([label]) => label === "Transport velocity extension into the sparse air band");
  const targetConfiguration = (sceneName === "ocean" || sceneName === "ocean-seiche")
    && brickFineResolution === 16 && presentationPageResolution === 16;
  const nonPressureGate = {
    threshold_ms: maximumTarget_ms,
    statistic: "p95 per-frame sum",
    minimumSamples: 24,
    scope: "every hardware-timestamped GPU phase except pressure-solve",
    requiredConfiguration: "ocean-seiche / Sparse CM12 / B16 / P16",
    eligible: targetConfiguration && sampled >= 24,
    passed: targetConfiguration && sampled >= 24 && nonPressureP95_ms < maximumTarget_ms,
  };
  const sharpeningCellAuthority = await qaSolver.readSharpeningCellAuthorityQA();
  const report = {
    probe: "sparse-cm12-stage-cost", scene: sceneName, samples: seen,
    diagnostic: {
      purpose: "FCA1/SRR1/VEX1 observability; not performance acceptance",
      passed: diagnosticFailure === undefined,
      failure: diagnosticFailure,
      initial: {
        frameControl: initialFrameControl,
        scalarAuthority: initialScalarAuthority,
      },
      authoritySamples,
      firstAuthorityFailure,
    },
    configuration: {
      brickFineResolution,
      presentationPageResolution,
      transportExperiment,
      refinementRegion: minimumCellSize === 0 ? undefined : {
        scope: regionScope,
        minimumCellSize_cells: minimumCellSize,
        bounds_m: scene.fluid.refinementRegions?.[0] === undefined ? undefined : {
          minimum: scene.fluid.refinementRegions[0].min_m,
          maximum: scene.fluid.refinementRegions[0].max_m,
        },
      },
      finestGrid: [solver.info.nx, solver.info.ny, solver.info.nz],
      dt_s,
      captureGap_ms,
      measurementSource: "gpu-hardware-timestamp",
    },
    workShape,
    scalarEventWorkShape: {
      count: scalarIngressEvents.length,
      yTileHistogram: scalarEventYHistogram,
    },
    adaptiveRepresentation,
    provenance: {
      gitCommit,
      gitDirty: gitStatus.trim().length > 0,
      gitStatusSha256: createHash("sha256").update(gitStatus).digest("hex"),
      backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
      methodProfile: "balanced",
      resolvedMethodValues: values,
      ladder: {
        productionBrickFineResolutions: [4, 8, 16],
        constructionQABrickFineResolutions: [4, 8, 16],
        brickFineResolution,
        presentationPageResolution,
      },
      pinnedReference: {
        path: PINNED_BASELINE_PATH,
        sha256: PINNED_BASELINE_SHA256,
        capturedAtLocal: "2026-08-17 08:05:00 Pacific/Auckland",
        status: "pre-final-boundary-fix reference; rerun before numerical comparison",
      },
    },
    medianAdvance_ms: Number(total.toFixed(4)),
    wallFrame: {
      samples_ms: wallFrameSamples.map((value) => Number(value.toFixed(4))),
      median_ms: Number(median(wallFrameSamples).toFixed(4)),
      p95_ms: Number(percentile(wallFrameSamples, 0.95).toFixed(4)),
    },
    pressureSolve_ms: Number(pressure.toFixed(4)),
    nonPressure_ms: Number(nonPressureMedian_ms.toFixed(4)),
    nonPressure: {
      samples_ms: nonPressureSamples.map((value) => Number(value.toFixed(4))),
      median_ms: Number(nonPressureMedian_ms.toFixed(4)),
      p95_ms: Number(nonPressureP95_ms.toFixed(4)),
      maximum_ms: Number(Math.max(...nonPressureSamples).toFixed(4)),
      cpuMedian_ms: Number(median(cpuNonPressureSamples).toFixed(4)),
      gate: nonPressureGate,
    },
    committedBricksPerFrame: committedSamples,
    pressureTopologyInputChangedPerFrame: pressureTopologyInputChangedSamples,
    pressureTopologyWork: pressureTopologyWorkSamples,
    pressureCutoverReceiptGate: {
      enforced: enforcePressureReceipts,
      completeSamples: pressureTopologyWorkSamples.filter(
        (sample) => sample.pressureAuthorityReceiptComplete).length,
      requiredSamples: sampled,
      passed: pressureTopologyWorkSamples.length === sampled
        && pressureTopologyWorkSamples.every((sample) =>
          sample.pressureAuthorityReceiptComplete),
      attribution: "prior accepted topology receipt; never current end-frame commit",
    },
    quiescentFrames: committedSamples.filter((value) => value === 0).length,
    phases: [...phaseSamples].map(([label, samples]) => ({
      label, median_ms: Number(median(samples).toFixed(4)), samples: samples.length,
    })).sort((left, right) => right.median_ms - left.median_ms),
    cpuPhases: [...cpuPhaseSamples].map(([label, samples]) => ({
      label, median_ms: Number(median(samples).toFixed(4)), samples: samples.length,
    })).sort((left, right) => right.median_ms - left.median_ms),
    cpuStages: [...cpuStageSamples].map(([id, samples]) => ({
      stage: id, median_ms: Number(median(samples).toFixed(4)), samples: samples.length,
    })).sort((left, right) => right.median_ms - left.median_ms),
    optimizationTarget: {
      stageIds: TARGET_STAGE_IDS,
      includesGammaDiffusion: true,
      samples_ms: targetSamples.map((value) => Number(value.toFixed(4))),
      median_ms: Number(targetMedian_ms.toFixed(4)),
      p95_ms: Number(targetP95_ms.toFixed(4)),
      maximum_ms: Number(Math.max(...targetSamples).toFixed(4)),
      cpuMedian_ms: Number(median(cpuTargetSamples).toFixed(4)),
    },
    frameAuthority: {
      abi: "FCA1",
      scheduling: "GPU-owned fixed indirect work/no-work families",
      // FCA1 is intentionally coalesced into the velocity-extension phase;
      // this is a conservative phase upper bound, not a fabricated isolated
      // timing for four singleton authority dispatches and one device copy.
      transportVelocityExtensionUpperBound_ms: frameAuthorityStage?.median_ms,
      cpuTransportVelocityExtensionUpperBound_ms: frameAuthorityCPU
        ? Number(median(frameAuthorityCPU[1]).toFixed(4)) : undefined,
      hostFluidAuthority: finalInfo?.hostFluidAuthority,
      hostSimulationSizedWorkItems: finalInfo?.hostSimulationSizedWorkItems,
      hostSchedulingUsesReadback: finalInfo?.hostSchedulingUsesReadback,
    },
    scalarAuthority: {
      abi: "SRR1",
      scheduling: "producer-authored result receipts plus GPU indirect tile ranks",
      ...lastScalarAuthorityHeader,
    },
    sharpeningCellAuthority: {
      abi: "SCA1",
      scheduling: "producer-authored air-side SRR1 work packets",
      ...sharpeningCellAuthority,
    },
    closure: {
      maximumAbsoluteError_ms: Math.max(...closureErrors.map(Math.abs)),
      exactWithin_ms: 1e-6,
    },
    terminalWork: finalInfo ? {
      acceptedCells: finalInfo.adaptiveAcceptedCellCount,
      acceptedRows: finalInfo.adaptiveAcceptedRowCount,
      pressureCells: finalInfo.adaptivePressureCellCount,
      pressureActiveRows: finalInfo.adaptivePressureActiveRowCount,
      temporalScalarCells: finalInfo.adaptiveTemporalScalarCellCount,
      temporalScalarRows: finalInfo.adaptiveTemporalScalarRowCount,
      temporalScalarRejectionMask: finalInfo.adaptiveTemporalScalarRejectionMask,
      residentBricks: finalInfo.fluidBrickResidentCount,
      topologyPrepared: finalInfo.adaptiveTopologyPreparedBrickCount,
      topologyCommitted: finalInfo.adaptiveTopologyCommittedBrickCount,
      pressureTopologyRepair: finalInfo.adaptivePressureTopologyRepair,
    } : undefined,
    stages, validationErrors,
  };
  console.log(JSON.stringify(report, null, 2));
  if (outputPath) await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  assert.equal(diagnosticFailure, undefined, diagnosticFailure);
  assert.deepEqual(validationErrors, []);
  if (enforceNonPressureGate) {
    assert.ok(nonPressureGate.eligible,
      `non-pressure gate requires ${nonPressureGate.requiredConfiguration} and at least ${
        nonPressureGate.minimumSamples} samples`);
    assert.ok(nonPressureGate.passed,
      `non-pressure p95 ${nonPressureP95_ms} ms is not below ${maximumTarget_ms} ms`);
  }
  if (enforcePressureReceipts) {
    const incomplete = pressureTopologyWorkSamples.filter(
      (sample) => !sample.pressureAuthorityReceiptComplete);
    assert.equal(incomplete.length, 0,
      `pressure cutover receipts missing/faulted: ${JSON.stringify(incomplete)}`);
  }
} finally {
  if (device) {
    const manager = gpuCompilationManagerFor(device);
    let teardownFailure: unknown;
    try {
      try {
        await manager.whenIdle();
        await device.queue.onSubmittedWorkDone();
      } catch (error) {
        teardownFailure = error;
      }
      try {
        teardownSolver?.destroy();
      } catch (error) {
        teardownFailure ??= error;
      } finally {
        teardownSolver = undefined;
      }
      invalidateGPUCompilationManager(device, "Sparse CM12 stage-cost probe complete");
      try {
        await manager.whenIdle();
        await device.queue.onSubmittedWorkDone();
      } catch (error) {
        teardownFailure ??= error;
      }
    } finally {
      device.destroy();
      // Let Dawn's retained ProcessEvents callbacks observe the drained native
      // device before this isolated process exits.
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
    if (teardownFailure) throw teardownFailure;
  }
  await releaseWebGPUExclusiveLock();
}
