#!/usr/bin/env node
/**
 * Construction-only paired pressure-addressing probe.
 *
 * The two required factories are deliberately not replaced by createAsync:
 * until the resident serial handoff installs both immutable QA constructors,
 * this tool fails before acquiring WebGPU.  Production method values, URL
 * state, scene state, and GPU state can never select either arm.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createOceanSeicheScene, createSymmetricExpansionScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { GPUStageTimestampRecorder } from "../lib/core/performance-trace";
import { usePerformanceInstrumentationStore } from
  "../lib/core/stores/performance-instrumentation-store";
import { resolveMethodValues } from "../lib/core/method-contract";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  type SparseCM12PressureAddressingABModeName,
  type SparseCM12PressureAddressingABReceipt,
  sparseCM12PressureAddressingABReceiptAccepted,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-addressing-ab";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";
import { inspectSparseCM12PressureCutoverAuthorities } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-cutover-observability";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;
const positive = (name: string, fallback: number): number => {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be positive`);
  return value;
};
const nonnegative = (name: string, fallback: number): number => {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be nonnegative`);
  }
  return value;
};

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(`Sparse CM12 construction-only pressure-addressing A/B

Usage:
  WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \\
  FLUID_WEBGPU_BACKEND=metal \\
  node --import tsx tools/probe-sparse-cm12-pressure-addressing-ab.ts [options]

Options:
  --scene=ocean-seiche             ocean-seiche or symmetric-expansion
  --brick-fine=16                  Fixed production brick ladder
  --presentation-page=16           Fixed production presentation page
  --warmup=N                       ocean: exactly 8; symmetric: exactly 0
  --frames=N                       ocean: exactly 24; symmetric: 5, 20, or 60
  --arm-order=rank-first           rank-first or list-first (default rank-first)
  --capture-gap-ms=N               Timestamp spacing (default 110)
  --require-bit-exact=0|1          Compare every measured physical frame (default 1)
  --enforce-pressure-receipts=0|1  Require accepted PAB1 list receipts (default 1)
  --out=PATH                       JSON artifact

Both arms require separate immutable QA constructors. There is no production
or run-time address-mode switch and no fallback when the list receipt faults.`);
  process.exit(0);
}

const sceneName = argument("scene", "ocean-seiche");
const brickFine = positive("brick-fine", 16);
const presentationPage = positive("presentation-page", 16);
const warmup = nonnegative("warmup", sceneName === "symmetric-expansion" ? 0 : 8);
const frames = positive("frames", sceneName === "symmetric-expansion" ? 5 : 24);
type PressureAddressingArmOrder = "rank-first" | "list-first";
const armOrderValue = argument("arm-order", "rank-first");
if (armOrderValue !== "rank-first" && armOrderValue !== "list-first") {
  throw new RangeError("arm-order must be rank-first or list-first");
}
const armOrder: PressureAddressingArmOrder = armOrderValue;
const captureGap_ms = positive("capture-gap-ms", 110);
const requireBitExact = argument("require-bit-exact", "1") === "1";
const enforcePressureReceipts = argument("enforce-pressure-receipts", "1") === "1";
const out = resolve(argument("out",
  sceneName === "symmetric-expansion"
    ? `artifacts/sparse-cm12-pressure-addressing-ab-symmetric${frames}.json`
    : "artifacts/sparse-cm12-pressure-addressing-ab-ocean24.json"));
if ((sceneName !== "ocean-seiche" && sceneName !== "symmetric-expansion")
  || brickFine !== 16 || presentationPage !== 16) {
  throw new Error("PAB1 supports only ocean-seiche or symmetric-expansion at B16/P16");
}
if (sceneName === "ocean-seiche" && (warmup !== 8 || frames !== 24)) {
  throw new Error("PAB1 ocean acceptance is fixed to warmup 8 / frames 24");
}
if (sceneName === "symmetric-expansion"
  && (warmup !== 0 || ![5, 20, 60].includes(frames))) {
  throw new Error("PAB1 symmetric correctness ladder requires warmup 0 and frames 5, 20, or 60");
}

const REPOSITORY_ROOT = fileURLToPath(new URL("..", import.meta.url));
const gitCommit = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: REPOSITORY_ROOT, encoding: "utf8",
}).trim();
const gitStatus = execFileSync("git", ["status", "--porcelain", "--untracked-files=all"], {
  cwd: REPOSITORY_ROOT, encoding: "utf8",
});

interface PartialFailureFrame {
  readonly arm: SparseCM12PressureAddressingABModeName;
  readonly phase: "construction" | "frame";
  readonly step: number;
  readonly fca: unknown;
  readonly srr: unknown;
  readonly pcm: unknown;
  readonly pressureAttribution: unknown;
  readonly rawPressureAuthority: unknown;
  readonly psaGenerations: unknown;
  readonly pab?: unknown;
  readonly hardwareTracePoll?: Readonly<{
    attempts: number; elapsed_ms: number; deadline_ms: number;
    ready: boolean; priorSampleId: number; observedSampleId?: number;
    observedSource?: string; expectedContext: string; observedContext?: string;
    enforcedInterAdvanceWait_ms: number;
  }>;
}
const partialFailureFrames: PartialFailureFrame[] = [];
let latestFailureFrame: PartialFailureFrame | undefined;
const rememberFailureFrame = (frame: PartialFailureFrame) => {
  latestFailureFrame = frame;
  partialFailureFrames.push(frame);
};

interface PressureAddressingFrameQA {
  readonly mode: SparseCM12PressureAddressingABModeName;
  readonly receipt?: SparseCM12PressureAddressingABReceipt;
  /** Separate hardware intervals; verification is never hidden in materialization. */
  readonly materialization_ms?: number;
  readonly verification_ms?: number;
}
type PressureAddressingSolver = WebGPUAdaptiveMassSolver & {
  readPressureAddressingABQA(): Promise<PressureAddressingFrameQA>;
};
type QAFactory = (
  device: GPUDevice,
  scene: ReturnType<typeof createOceanSeicheScene>,
  quality: "balanced",
  onRigidLoads: undefined,
  options: ReturnType<typeof adaptiveMassSolverOptions>,
  onProgress: () => void,
  signal?: AbortSignal,
) => Promise<PressureAddressingSolver>;
type QAConstructors = {
  createPressureAddressingRankSelectForQA?: QAFactory;
  createPressureAddressingMaterializedListForQA?: QAFactory;
};
const constructors = WebGPUAdaptiveMassSolver as unknown as QAConstructors;
const rankFactory = constructors.createPressureAddressingRankSelectForQA;
const listFactory = constructors.createPressureAddressingMaterializedListForQA;
if (!rankFactory || !listFactory) {
  throw new Error("PAB1 resident integration gap: install both construction-only factories "
    + "createPressureAddressingRankSelectForQA and "
    + "createPressureAddressingMaterializedListForQA; createAsync fallback is forbidden");
}

const median = (values: readonly number[]): number => {
  assert(values.length > 0);const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.floor((ordered.length - 1) / 2)]!;
};
const p95 = (values: readonly number[]): number => {
  assert(values.length > 0);const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.ceil(0.95 * ordered.length) - 1]!;
};
const hashView = (view: ArrayBufferView): string => createHash("sha256")
  .update(new Uint8Array(view.buffer, view.byteOffset, view.byteLength)).digest("hex");

interface ArmFrame {
  readonly step: number;
  readonly fields?: Readonly<Record<string, string>>;
  readonly receipt?: SparseCM12PressureAddressingABReceipt;
  readonly materialization_ms: number;
  readonly verification_ms: number;
  readonly pressureRhs_ms: number;
  readonly pressureSolve_ms: number;
  readonly srr: Readonly<Record<string, number>>;
  readonly pcm: Readonly<Record<string, number | string | boolean>>;
  readonly pressureReceipts: { readonly complete: boolean;
    readonly issues: readonly string[] };
  readonly hardwareTracePoll: NonNullable<PartialFailureFrame["hardwareTracePoll"]>;
  readonly symmetry?: Readonly<{
    densityD4: number; velocityD4_m_s: number; pressureD4: number;
    divergenceD4_s: number;
  }>;
}
interface ArmResult {
  readonly mode: SparseCM12PressureAddressingABModeName;
  readonly frames: readonly ArmFrame[];
}

const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
  timeStep: sceneName === "symmetric-expansion" ? "paper" : "scene",
  brickFineResolution: "16", presentationPageResolution: "16",
});
const options = adaptiveMassSolverOptions(values);
const provenance = Object.freeze({
  gitCommit,
  gitDirty: gitStatus.trim().length > 0,
  gitStatusSha256: createHash("sha256").update(gitStatus).digest("hex"),
  backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
  methodProfile: "balanced",
  resolvedMethodValues: values,
  ladder: Object.freeze({
    supportedBrickFineResolutions: [16],
    brickFineResolution: brickFine,
    presentationPageResolution: presentationPage,
    symmetricCorrectnessFrameRungs: [5, 20, 60],
    selectedFrameRung: frames,
  }),
});
const pressureRhsLabel = "Finite-volume divergence RHS + compatibility projection";
const pressureSolveLabel = "One-reduction sparse MGPCG pressure solve";

/** Exact scene construction shared with the weakened temporal-regression lane. */
const createWeakenedSymmetricScene = () => {
  const dimensions = [32, 16, 32] as const;
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
  return scene;
};
const createProbeScene = () => sceneName === "symmetric-expansion"
  ? createWeakenedSymmetricScene() : createOceanSeicheScene();

const scalarD4Error = (field: ArrayLike<number>, dimensions: readonly number[]): number => {
  const [nx, ny, nz] = dimensions as readonly [number, number, number];
  let maximum = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1) for (const [tx, tz] of [
      [nx - 1 - x, z], [x, nz - 1 - z], [z, x],
    ] as const) {
      const source = Number(field[x + nx * (y + ny * z)]);
      const target = Number(field[tx + nx * (y + ny * tz)]);
      assert(Number.isFinite(source) && Number.isFinite(target),
        "symmetric scalar publication contains non-finite values");
      maximum = Math.max(maximum, Math.abs(source - target));
    }
  return maximum;
};

const velocityD4Error = (field: ArrayLike<number>, dimensions: readonly number[]): number => {
  const [nx, ny, nz] = dimensions as readonly [number, number, number];
  let maximum = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1)
      for (const transform of ["reflect-x", "reflect-z", "swap-xz"] as const) {
        const target = transform === "reflect-x" ? [nx - 1 - x, y, z]
          : transform === "reflect-z" ? [x, y, nz - 1 - z] : [z, y, x];
        const sourceCell = x + nx * (y + ny * z);
        const targetCell = target[0]! + nx * (target[1]! + ny * target[2]!);
        for (let component = 0; component < 3; component += 1) {
          const targetComponent = transform === "swap-xz"
            ? component === 0 ? 2 : component === 2 ? 0 : 1 : component;
          const source = Number(field[4 * sourceCell + component]);
          const expected = transform === "reflect-x" && component === 0 ? -source
            : transform === "reflect-z" && component === 2 ? -source : source;
          const actual = Number(field[4 * targetCell + targetComponent]);
          assert(Number.isFinite(expected) && Number.isFinite(actual),
            "symmetric velocity publication contains non-finite values");
          maximum = Math.max(maximum, Math.abs(expected - actual));
        }
      }
  return maximum;
};

async function runArm(device: GPUDevice, mode: SparseCM12PressureAddressingABModeName,
  factory: QAFactory): Promise<ArmResult> {
  const scene = createProbeScene();
  const solver = await factory.call(WebGPUAdaptiveMassSolver,
    device, scene, "balanced", undefined, options, () => {});
  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
  const samples: ArmFrame[] = [];
  let priorSampleId = 0;
  try {
    // Prime the readback-only attribution tracker with the construction
    // terminal. This is not a physics warmup: no advance is encoded. Without
    // it, a first read after step 1 is deliberately marked unavailable because
    // the tracker cannot infer the prior pressure input generation.
    const [constructionStats, constructionFCA, constructionSRR, constructionPCM] =
      await Promise.all([solver.readStats(), solver.readFrameControlQA(),
        solver.readScalarAuthorityQA(), solver.readPressureCanonicalMembershipQA()]);
    const constructionAttribution = constructionStats.adaptivePressureTopologyAttribution;
    rememberFailureFrame({ arm: mode, phase: "construction", step: 0,
      fca: constructionFCA, srr: constructionSRR,
      pcm: { cell: constructionPCM.cell, row: constructionPCM.row },
      pressureAttribution: constructionAttribution ?? null,
      rawPressureAuthority: constructionAttribution?.authorities ?? null,
      psaGenerations: constructionAttribution?.authorities?.psa ? {
        acceptedGeneration: constructionAttribution.authorities.psa.acceptedGeneration,
        candidateGeneration: constructionAttribution.authorities.psa.candidateGeneration,
      } : null });
    assert(constructionAttribution,
      `${mode} construction pressure attribution is absent`);
    assert.equal(constructionAttribution.status, "matched",
      `${mode} construction pressure attribution is not matched`);
    assert.equal(constructionAttribution.encodedStep, 0,
      `${mode} construction pressure attribution is not step 0`);
    assert.equal(constructionAttribution.inputTopologyGeneration,
      constructionAttribution.currentEndFrameTopologyGeneration,
      `${mode} construction pressure attribution input generation is stale`);
    for (let step = 1; step <= warmup + frames; step += 1) {
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
      await device.queue.onSubmittedWorkDone();
      const pollStarted_ms = performance.now();
      const expectedContext = `adaptive-mass:sim-${(step * dt_s).toFixed(6)}`;
      let attempts = 0;
      let stats: Awaited<ReturnType<PressureAddressingSolver["readStats"]>>;
      let traceReady = false;
      do {
        attempts += 1;
        stats = await solver.readStats();
        const candidate = stats.physicsTrace;
        traceReady = candidate?.measurementSource === "gpu-hardware-timestamp"
          && candidate.sampleId > priorSampleId
          && candidate.context === expectedContext;
        if (traceReady) break;
        const remaining_ms = captureGap_ms - (performance.now() - pollStarted_ms);
        if (remaining_ms <= 0) break;
        await new Promise((done) => setTimeout(done, Math.min(5, remaining_ms)));
      } while (true);
      const hardwareTracePoll = { attempts,
        elapsed_ms: performance.now() - pollStarted_ms,
        deadline_ms: captureGap_ms, ready: traceReady, priorSampleId, expectedContext,
        enforcedInterAdvanceWait_ms: 0,
        ...(stats.physicsTrace ? { observedSampleId: stats.physicsTrace.sampleId,
          observedSource: stats.physicsTrace.measurementSource,
          observedContext: stats.physicsTrace.context } : {}) };
      const [qa, fcaQA, srrQA, pcmQA] = await Promise.all([
        solver.readPressureAddressingABQA(),
        solver.readFrameControlQA(), solver.readScalarAuthorityQA(),
        solver.readPressureCanonicalMembershipQA(),
      ]);
      const srr = Object.freeze({ phase: srrQA.phase, fault: srrQA.fault,
        acceptedGeneration: srrQA.acceptedGeneration,
        candidateGeneration: srrQA.candidateGeneration,
        scheduledWork: srrQA.scheduledWork, executedWork: srrQA.executedWork });
      const pcm = Object.freeze({
        cellPhase: pcmQA.cell.phase, cellFault: pcmQA.cell.fault,
        cellAcceptedGeneration: pcmQA.cell.acceptedGeneration,
        cellCandidateGeneration: pcmQA.cell.candidateGeneration,
        cellTotalCount: pcmQA.cell.totalCount, cellActiveBitCount: pcmQA.cell.activeBitCount,
        cellActiveBitsSha256: pcmQA.cell.activeBitsSha256,
        rowPhase: pcmQA.row.phase, rowFault: pcmQA.row.fault,
        rowAcceptedGeneration: pcmQA.row.acceptedGeneration,
        rowCandidateGeneration: pcmQA.row.candidateGeneration,
        rowTotalCount: pcmQA.row.totalCount, rowActiveBitCount: pcmQA.row.activeBitCount,
        rowActiveBitsSha256: pcmQA.row.activeBitsSha256,
      });
      const pressureAttribution = stats.adaptivePressureTopologyAttribution;
      const pressureReceipts = inspectSparseCM12PressureCutoverAuthorities(
        pressureAttribution?.authorities,
        pressureAttribution?.inputTopologyGeneration);
      rememberFailureFrame({ arm: mode, phase: "frame", step,
        fca: fcaQA, srr: srrQA, pcm: { cell: pcmQA.cell, row: pcmQA.row },
        pressureAttribution: pressureAttribution ?? null,
        rawPressureAuthority: pressureAttribution?.authorities ?? null,
        psaGenerations: pressureAttribution?.authorities?.psa ? {
          acceptedGeneration: pressureAttribution.authorities.psa.acceptedGeneration,
          candidateGeneration: pressureAttribution.authorities.psa.candidateGeneration,
        } : null,
        pab: qa, hardwareTracePoll });
      assert.equal(qa.mode, mode, `${mode} constructor published ${qa.mode}`);
      const trace = stats.physicsTrace;
      if (step <= warmup) {
        if (traceReady) priorSampleId = trace!.sampleId;
        const remainingGap_ms = captureGap_ms - (performance.now() - pollStarted_ms);
        hardwareTracePoll.enforcedInterAdvanceWait_ms = Math.max(0, remainingGap_ms);
        if (remainingGap_ms > 0) {
          await new Promise((done) => setTimeout(done, remainingGap_ms));
        }
        continue;
      }
      assert(trace && trace.measurementSource === "gpu-hardware-timestamp",
        `${mode} frame ${step} has no hardware timestamp trace`);
      assert(trace.sampleId > priorSampleId, `${mode} reused timestamp sample ${trace.sampleId}`);
      assert.equal(trace.context, expectedContext,
        `${mode} frame ${step} received stale hardware context ${trace.context}`);
      priorSampleId = trace.sampleId;
      const phase = (label: string) => trace.phases.find((entry) => entry.label === label)
        ?.duration_ms ?? Number.NaN;
      const pressureRhs_ms = phase(pressureRhsLabel);
      const pressureSolve_ms = phase(pressureSolveLabel);
      assert(Number.isFinite(pressureRhs_ms) && Number.isFinite(pressureSolve_ms),
        `${mode} frame ${step} is missing pressure phase timestamps`);
      if (mode === "materializedList" && enforcePressureReceipts) {
        assert(qa.receipt && sparseCM12PressureAddressingABReceiptAccepted(qa.receipt),
          `${mode} frame ${step} has no accepted PAB1 receipt`);
      }
      if (enforcePressureReceipts) assert(pressureReceipts.complete,
        `${mode} frame ${step} pressure receipts incomplete: ${pressureReceipts.issues.join("; ")}`);
      let fields: Readonly<Record<string, string>> | undefined;
      let symmetry: ArmFrame["symmetry"];
      if (requireBitExact) {
        const diagnostic = await solver.readDiagnosticFields();
        fields = Object.freeze(Object.fromEntries(Object.entries(diagnostic).map(
          ([name, view]) => [name, hashView(view as ArrayBufferView)])));
        if (sceneName === "symmetric-expansion") {
          const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
          assert.deepEqual(dimensions, [32, 16, 32], `${mode} symmetric grid mismatch`);
          symmetry = Object.freeze({
            densityD4: scalarD4Error(diagnostic.density, dimensions),
            velocityD4_m_s: velocityD4Error(diagnostic.velocity, dimensions),
            pressureD4: scalarD4Error(diagnostic.pressure, dimensions),
            divergenceD4_s: scalarD4Error(diagnostic.divergence, dimensions),
          });
          assert(symmetry.densityD4 <= 5e-3,
            `${mode} frame ${step} density D4 ${symmetry.densityD4} exceeds 0.005`);
          assert(symmetry.velocityD4_m_s <= 2e-3,
            `${mode} frame ${step} velocity D4 ${symmetry.velocityD4_m_s} exceeds 0.002`);
          assert(symmetry.pressureD4 <= 1,
            `${mode} frame ${step} pressure D4 ${symmetry.pressureD4} exceeds 1`);
        }
      }
      if (step < warmup + frames) {
        const remainingGap_ms = captureGap_ms - (performance.now() - pollStarted_ms);
        hardwareTracePoll.enforcedInterAdvanceWait_ms = Math.max(0, remainingGap_ms);
        if (remainingGap_ms > 0) {
          await new Promise((done) => setTimeout(done, remainingGap_ms));
        }
      }
      samples.push({ step, fields, receipt: qa.receipt,
        materialization_ms: qa.materialization_ms ?? 0,
        verification_ms: qa.verification_ms ?? 0,
        pressureRhs_ms, pressureSolve_ms, srr, pcm, pressureReceipts,
        hardwareTracePoll: Object.freeze({ ...hardwareTracePoll }),
        ...(symmetry ? { symmetry } : {}) });
    }
  } finally {
    // Let GPUCompilationManager-owned work and all submitted readbacks retire
    // before releasing a resident; this avoids Dawn pipeline/device UAF during
    // the serial handoff between immutable arms.
    await device.queue.onSubmittedWorkDone();
    solver.destroy();
  }
  assert.equal(samples.length, frames);
  return Object.freeze({ mode, frames: Object.freeze(samples) });
}

usePerformanceInstrumentationStore.getState().setMode("timeline");
await acquireWebGPUExclusiveLock("pressure-addressing-ab", `PAB1 ${sceneName} B16/P16 paired probe`);
let device: GPUDevice | undefined;
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert(adapter, "PAB1 requires a WebGPU adapter");
  assert(adapter.features.has("timestamp-query"), "PAB1 requires hardware timestamp queries");
  device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query" as GPUFeatureName],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  await GPUStageTimestampRecorder.prepare(device);
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();validationErrors.push(event.error.message);
  });
  // Sequential arms keep the scene construction footprint bounded. Both start
  // from a new deterministic scene and execute exactly the same step sequence.
  // Arm order is a construction-only probe input so complementary artifacts can
  // expose DVFS/thermal/order bias without creating a production address switch.
  const orderedArms = armOrder === "rank-first" ? [
    ["canonicalRankSelect", rankFactory], ["materializedList", listFactory],
  ] as const : [
    ["materializedList", listFactory], ["canonicalRankSelect", rankFactory],
  ] as const;
  const first = await runArm(device, orderedArms[0][0], orderedArms[0][1]);
  const second = await runArm(device, orderedArms[1][0], orderedArms[1][1]);
  const rank = first.mode === "canonicalRankSelect" ? first : second;
  const list = first.mode === "materializedList" ? first : second;
  assert.equal(rank.mode, "canonicalRankSelect");
  assert.equal(list.mode, "materializedList");
  assert.deepEqual(validationErrors, [], "PAB1 WebGPU validation errors");
  if (requireBitExact) for (let frame = 0; frame < frames; frame += 1) {
    assert.deepEqual(list.frames[frame]!.fields, rank.frames[frame]!.fields,
      `PAB1 physical fields differ at measured frame ${frame + 1}`);
  }
  for (let frame = 0; frame < frames; frame += 1) {
    assert.deepEqual(list.frames[frame]!.srr, rank.frames[frame]!.srr,
      `PAB1 SRR common-mode state differs at measured frame ${frame + 1}`);
    assert.deepEqual(list.frames[frame]!.pcm, rank.frames[frame]!.pcm,
      `PAB1 PCM common-mode state differs at measured frame ${frame + 1}`);
    assert.deepEqual(list.frames[frame]!.pressureReceipts,
      rank.frames[frame]!.pressureReceipts,
      `PAB1 pressure receipts differ at measured frame ${frame + 1}`);
    assert.deepEqual(list.frames[frame]!.symmetry, rank.frames[frame]!.symmetry,
      `PAB1 symmetric physical receipt differs at measured frame ${frame + 1}`);
  }
  const valuesFor = (arm: ArmResult, key: keyof Pick<ArmFrame,
    "materialization_ms" | "verification_ms" | "pressureRhs_ms" | "pressureSolve_ms">) =>
    arm.frames.map((frame) => frame[key]);
  const summarize = (samples: readonly number[]) => ({
    samples_ms: samples, median_ms: median(samples), p95_ms: p95(samples),
  });
  const allSRRFaultZero = [...rank.frames, ...list.frames]
    .every((frame) => frame.srr.fault === 0);
  const symmetricCorrectnessOnly = sceneName === "symmetric-expansion";
  const artifact = {
    passed: true, scene: sceneName,
    configuration: { brickFineResolution: brickFine,
      presentationPageResolution: presentationPage, warmup, frames, captureGap_ms,
      measurementSource: "gpu-hardware-timestamp", sequentialArms: true,
      armOrder, counterbalanced: false },
    provenance,
    bitExactEveryMeasuredFrame: requireBitExact,
    // A single sequential artifact cannot distinguish addressing cost from
    // DVFS/thermal/cache order. Only an explicit combined/counterbalanced
    // analysis may promote these observations into a performance claim.
    performanceClaim: false,
    performanceValidity: symmetricCorrectnessOnly ? "correctness-only-no-performance-claim"
      : allSRRFaultZero ? "sequential-order-confounded" : "common-mode-qa-only",
    performanceValidityReason: symmetricCorrectnessOnly
      ? "symmetric 5/20/60 ladder artifacts certify correctness and authority only"
      : allSRRFaultZero
        ? `${armOrder} is one sequential arm order; combine it with the complementary order before claiming a speedup`
        : "SRR fault persisted identically in both arms; address delta is QA-only",
    commonModeReceipts: {
      physicalHashesIdenticalEveryMeasuredFrame: requireBitExact,
      symmetryReceiptsIdenticalEveryMeasuredFrame: sceneName === "symmetric-expansion",
      srrIdenticalEveryMeasuredFrame: true,
      pcmIdenticalEveryMeasuredFrame: true,
      pressureReceiptsCompleteEveryMeasuredFrame: true,
      rank: rank.frames.map(({ step, srr, pcm, pressureReceipts }) =>
        ({ step, srr, pcm, pressureReceipts })),
      materializedList: list.frames.map(({ step, srr, pcm, pressureReceipts }) =>
        ({ step, srr, pcm, pressureReceipts })),
    },
    physicalReceipts: {
      rank: rank.frames.map(({ step, fields, symmetry, hardwareTracePoll }) =>
        ({ step, fields, symmetry, hardwareTracePoll })),
      materializedList: list.frames.map(
        ({ step, fields, symmetry, hardwareTracePoll }) =>
          ({ step, fields, symmetry, hardwareTracePoll })),
    },
    rankSelect: {
      pressureRhs: summarize(valuesFor(rank, "pressureRhs_ms")),
      pressureSolve: summarize(valuesFor(rank, "pressureSolve_ms")),
    },
    materializedList: {
      materialization: summarize(valuesFor(list, "materialization_ms")),
      verification: summarize(valuesFor(list, "verification_ms")),
      pressureRhs: summarize(valuesFor(list, "pressureRhs_ms")),
      pressureSolve: summarize(valuesFor(list, "pressureSolve_ms")),
      receipts: list.frames.map((frame) => frame.receipt),
    },
    solveMedianDelta_ms: median(valuesFor(list, "pressureSolve_ms"))
      - median(valuesFor(rank, "pressureSolve_ms")),
    solvePlusMaterializationMedianDelta_ms:
      median(list.frames.map((frame) => frame.pressureSolve_ms + frame.materialization_ms))
      - median(valuesFor(rank, "pressureSolve_ms")),
  };
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(artifact, null, 2)}\n`);
  console.log(JSON.stringify(artifact, null, 2));
} catch (error) {
  const failureArtifact = {
    passed: false, scene: sceneName,
    configuration: { brickFineResolution: brickFine,
      presentationPageResolution: presentationPage, warmup, frames, captureGap_ms,
      measurementSource: "gpu-hardware-timestamp", sequentialArms: true,
      armOrder, counterbalanced: false,
      constructionAttributionPrimedWithoutPhysicsAdvance: true },
    provenance,
    performanceClaim: false,
    performanceValidity: "failed-no-performance-claim",
    error: error instanceof Error ? { name: error.name, message: error.message,
      stack: error.stack } : { message: String(error) },
    latestFailureFrame: latestFailureFrame ?? null,
    partialFailureFrames,
  };
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(failureArtifact, null, 2)}\n`);
  console.error(JSON.stringify(failureArtifact, null, 2));
  throw error;
} finally {
  if (device) {
    try {
      await gpuCompilationManagerFor(device).whenIdle();
      await device.queue.onSubmittedWorkDone();
    } catch (error) {
      console.error("PAB1 GPU teardown fence failed", error);
    }
  }
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
