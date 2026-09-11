/**
 * Three-second production-default correctness run for every Sparse Geometric
 * complexity-ladder scene. Each scene owns a fresh Dawn process and the
 * repository WebGPU lease.
 */
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { sparseCM12DawnDefaultValues } from
  "../lib/harness/sparse-cm12-dawn-defaults";
import { resolveMethodValues } from "../lib/core/method-contract";
import { SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE,
  SPARSE_CM12_COMPLEXITY_SCENES, getScenePreset,
  type SparseCM12ComplexitySceneId } from "../lib/core/scenes";
import { gpuCompilationManagerFor, invalidateGPUCompilationManager,
  managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { SimulationFailureError } from "../lib/core/simulation-failure";
import { acquireWebGPUExclusiveLock, readWebGPUExclusiveLockHolder,
  releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { fingerprintSparseCM12RepositorySources,
  type SparseCM12SourceContentFingerprint } from
  "./sparse-cm12-source-content-fingerprint";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SELF = fileURLToPath(import.meta.url);
const DEFAULT_DAWN_MODULE = join(ROOT, "node_modules/webgpu/index.js");
const DEFAULT_DURATION_S = 3;
const DEFAULT_SCENE_TIMEOUT_MS = 30 * 60_000;
const BALANCE_RELATIVE_TOLERANCE = 0.005;

const argument = (name: string): string | undefined => {
  const inline = process.argv.slice(2).find(value => value.startsWith(`--${name}=`));
  if (inline) return inline.slice(name.length + 3);
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`--${name} requires a value`);
  return value;
};
const flag = (name: string) => process.argv.includes(`--${name}`);
const finiteNumber = (name: string, fallback: number): number => {
  const value = argument(name) === undefined ? fallback : Number(argument(name));
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`--${name} must be positive`);
  return value;
};

interface FieldReceipt {
  readonly passed: boolean;
  readonly values: number;
  readonly nonfiniteByField: Readonly<Record<string, number>>;
}

type AcceptedVolumeReceipt = Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readAcceptedGeometricVolumeQA"]>>;
type TransportReceipt = Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readGeometricVolumeTransportReceiptQA"]>>;

interface BalanceReceipt {
  readonly initialVolumeFine3: number;
  readonly finalVolumeFine3: number;
  readonly sourceEmittedFine3: number;
  readonly cumulativeOutflowFine3: number;
  readonly cumulativeOutflowQuantizationBoundFine3: number;
  readonly errorFine3: number;
  readonly relativeError: number;
  readonly maximumRelativeError: number;
  readonly passed: boolean;
}

interface SceneReceipt {
  readonly scene: SparseCM12ComplexitySceneId;
  readonly title: string;
  readonly methodValues: unknown;
  readonly passed: boolean;
  readonly durationRequested_s: number;
  readonly completedTime_s: number | null;
  readonly submittedTime_s: number | null;
  readonly encodedSteps: number | null;
  readonly initialAcceptedVolume: AcceptedVolumeReceipt | null;
  readonly acceptedVolume: AcceptedVolumeReceipt | null;
  readonly finiteFields: FieldReceipt | null;
  readonly balance: BalanceReceipt | null;
  /** Latest fully accepted frame. A rejected frame is reported separately. */
  readonly transport: TransportReceipt | null;
  readonly failureTransport: TransportReceipt | null;
  readonly rejectedFrame: {
    readonly accepted: false;
    readonly priorAcceptedTime_s: number;
    readonly attemptedCompletedTime_s: number | null;
    readonly physicalDt_s: number;
    readonly executedPhysicalDt_s: number;
    readonly plannedSubsteps: number;
    readonly executedSubsteps: number;
    readonly fault: number;
  } | null;
  readonly validationErrors: readonly string[];
  readonly deviceLost: string | null;
  readonly wall_ms: number;
  readonly failure: string | null;
  readonly sourceFingerprint: SparseCM12SourceContentFingerprint;
  readonly sourceFingerprintAfter: SparseCM12SourceContentFingerprint | null;
  readonly sourceUnchanged: boolean | null;
  readonly failureOwnerCellRows?: unknown;
  readonly failureOwnerRow?: unknown;
}

const sceneDefinition = (id: string) => {
  const found = SPARSE_CM12_COMPLEXITY_SCENES.find(scene => scene.id === id);
  if (!found) throw new Error(`Unknown Sparse Geometric ladder scene ${id}`);
  return found;
};

function inspectFields(fields: Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readDiagnosticFields"]>>): FieldReceipt {
  const nonfiniteByField: Record<string, number> = {};
  let values = 0;
  for (const [name, field] of Object.entries(fields)) {
    let nonfinite = 0;
    for (const value of field) nonfinite += Number(!Number.isFinite(value));
    nonfiniteByField[name] = nonfinite;
    values += field.length;
  }
  return { passed: Object.values(nonfiniteByField).every(count => count === 0),
    values, nonfiniteByField };
}

function requireCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function worker(): Promise<void> {
  const definition = sceneDefinition(process.env.FLUID_CM12_LADDER_CORRECTNESS_SCENE ?? "");
  const duration_s = Number(process.env.FLUID_CM12_LADDER_CORRECTNESS_DURATION_S);
  const output = process.env.FLUID_CM12_LADDER_CORRECTNESS_RECEIPT;
  if (!output || !Number.isFinite(duration_s) || duration_s <= 0) {
    throw new Error("Invalid internal ladder correctness worker configuration");
  }
  const methodValues = resolveMethodValues(adaptiveMassMethod,
    SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE.quality,
    SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE.overrides ?? {});
  requireCondition(JSON.stringify(methodValues) === JSON.stringify(
    sparseCM12DawnDefaultValues()),
  "Sparse Geometric ladder catalog profile differs from production defaults");
  const sourceFingerprint = await fingerprintSparseCM12RepositorySources(ROOT);
  await acquireWebGPUExclusiveLock("dawn-ladder-correctness",
    `Sparse Geometric 3 s ladder: ${definition.id}`);
  const started = performance.now();
  let sourceFingerprintAfter: SparseCM12SourceContentFingerprint | null = null;
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  let lost: GPUDeviceLostInfo | undefined;
  const validationErrors: string[] = [];
  let initialAcceptedVolume: AcceptedVolumeReceipt | null = null;
  let acceptedVolume: AcceptedVolumeReceipt | null = null;
  let finiteFields: FieldReceipt | null = null;
  let balance: BalanceReceipt | null = null;
  let transport: TransportReceipt | null = null;
  let failureTransport: TransportReceipt | null = null;
  let rejectedFrame: SceneReceipt["rejectedFrame"] = null;
  let completedTime_s: number | null = null;
  let submittedTime_s: number | null = null;
  let encodedSteps: number | null = null;
  let lastAcceptedTime_s = 0;
  let lastAcceptedSteps = 0;
  let failure: string | null = null;
  let failureOwnerCellRows: unknown;
  let failureOwnerRow: unknown;
  let initializationPhase = "initializing-dawn";
  const writeProgress = (phase: string, extra: Record<string, unknown> = {}) => {
    const info = solver?.info;
    const compilation = device ? gpuCompilationManagerFor(device).snapshot() : undefined;
    process.stderr.write(`${JSON.stringify({ phase, scene: definition.id,
      acceptedTime_s: lastAcceptedTime_s, acceptedSteps: lastAcceptedSteps,
      solverSubmittedTime_s: info?.submittedTime_s,
      solverEncodedSteps: info?.encodedSteps,
      topologyGenerationPending: info?.topologyGenerationPending,
      topologyGenerationDeferred: info?.topologyGenerationDeferred,
      framePending: solver?.framePending, compilation, ...extra })}\n`);
  };
  const initializationHeartbeat = setInterval(() => writeProgress(initializationPhase), 15_000);
  initializationHeartbeat.unref();
  const makeReceipt = (passed: boolean, receiptFailure: string | null): SceneReceipt => ({
    scene: definition.id, title: definition.title, methodValues, passed,
    durationRequested_s: duration_s, completedTime_s, submittedTime_s, encodedSteps,
    initialAcceptedVolume, acceptedVolume, finiteFields, balance, transport,
    failureTransport, rejectedFrame,
    validationErrors, deviceLost: lost ? `${lost.reason}: ${lost.message}` : null,
    wall_ms: Number((performance.now() - started).toFixed(1)), failure: receiptFailure,
    sourceFingerprint, sourceFingerprintAfter,
    sourceUnchanged: sourceFingerprintAfter === null ? null
      : sourceFingerprintAfter.sha256 === sourceFingerprint.sha256,
    ...(failureOwnerCellRows === undefined ? {} : { failureOwnerCellRows }),
    ...(failureOwnerRow === undefined ? {} : { failureOwnerRow }),
  });
  try {
    writeProgress(initializationPhase);
    requireCondition(adaptiveMassMethod.id === "adaptive-volume",
      "ladder runner resolved a non-geometric method");
    const modulePath = process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE;
    const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn,
      [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
        "enable-dawn-features=disable_blob_cache"]);
    Object.defineProperty(globalThis, "navigator", {
      configurable: true, value: { gpu },
    });
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    requireCondition(adapter, "Dawn did not expose a WebGPU adapter");
    device = managedGPUDevice(await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    }), { requireWorkerRealm: false, maximumConcurrentBundles: 1 });
    void device.lost.then(info => { lost = info; });
    device.addEventListener("uncapturederror", event => {
      event.preventDefault(); validationErrors.push(event.error.message);
    });
    initializationPhase = "creating-solver";
    writeProgress(initializationPhase);
    const scene = getScenePreset(`sparse-cm12-ladder-${definition.id}`).create();
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
      methodValues, undefined, () => {}) as WebGPUAdaptiveMassSolver;
    initializationPhase = "waiting-simulation-ready";
    writeProgress(initializationPhase);
    await solver.waitForSimulationReady();
    await device.queue.onSubmittedWorkDone();
    initialAcceptedVolume = await solver.readAcceptedGeometricVolumeQA();
    requireCondition(initialAcceptedVolume.invalidCells === 0,
      "initial accepted volume contains invalid cells");
    requireCondition(initialAcceptedVolume.nonfiniteDynamicsCells === 0,
      "initial accepted cells contain nonfinite dynamics");
    clearInterval(initializationHeartbeat);
    writeProgress("initialization-complete");

    let cumulativeOutflowFine3 = 0;
    let cumulativeOutflowQuantizationBoundFine3 = 0;
    let lastProgressAt = performance.now();
    const admissionTimeoutMs = Number(
      process.env.FLUID_CM12_LADDER_CORRECTNESS_ADMISSION_TIMEOUT_MS ?? 300_000);
    while (lastAcceptedTime_s < duration_s - 1e-9) {
      const deadline = performance.now() + admissionTimeoutMs;
      let lastAdmissionProgressAt = -Infinity;
      while (!solver.advanceTo(duration_s, [])) {
        const deferred = solver.info.topologyGenerationDeferred;
        if (deferred?.reason === "volume-capacity") {
          throw new Error(`topology generation refused required volume capacity: ${
            deferred.detail ?? JSON.stringify(deferred)}`);
        }
        if (lost) throw new Error(`device lost: ${lost.reason} ${lost.message}`);
        if (validationErrors.length > 0) {
          throw new Error(`Dawn validation: ${validationErrors.join("; ")}`);
        }
        requireCondition(performance.now() < deadline,
          `advance admission stalled before ${duration_s} s`);
        if (performance.now() - lastAdmissionProgressAt >= 15_000) {
          lastAdmissionProgressAt = performance.now();
          writeProgress("admission-wait", {
            admissionWait_ms: Math.round(admissionTimeoutMs - (deadline - performance.now())),
          });
        }
        await new Promise<void>(resolve => setImmediate(resolve));
      }
      writeProgress("frame-completion-wait");
      const frameHeartbeat = setInterval(() => writeProgress("frame-completion-wait"), 15_000);
      frameHeartbeat.unref();
      try { await solver.awaitFrameCompletion?.(); }
      finally { clearInterval(frameHeartbeat); }
      await device.queue.onSubmittedWorkDone();
      if (lost) throw new Error(`device lost: ${lost.reason} ${lost.message}`);
      requireCondition(validationErrors.length === 0,
        `Dawn validation: ${validationErrors.join("; ")}`);
      const frameTransport = await solver.readGeometricVolumeTransportReceiptQA();
      transport = frameTransport;
      requireCondition(frameTransport.fault === 0, `transport fault ${frameTransport.fault}`);
      requireCondition(frameTransport.transportCompleted,
        "transport did not complete its planned microsteps");
      const completedDt_s = solver.info.lastDt_s;
      requireCondition(Number.isFinite(completedDt_s) && completedDt_s! > 0,
        "completed frame omitted its physical dt");
      requireCondition(Math.abs(frameTransport.executedPhysicalDt_s - completedDt_s!)
        <= 8 * 2 ** -23 * completedDt_s!,
      "transport executed duration differs from the completed outer step");
      cumulativeOutflowFine3 += frameTransport.outflowFineCells3;
      cumulativeOutflowQuantizationBoundFine3 +=
        frameTransport.outflowQuantizationBoundFine3;
      // Keep an independent accepted-frame clock. Failure settlement leaves
      // solver.info describing the prior publication and some fault paths
      // report placeholder values, while the completed transport duration is
      // the authoritative increment for this successful frame.
      lastAcceptedTime_s = Math.min(duration_s, lastAcceptedTime_s + completedDt_s!);
      lastAcceptedSteps += 1;
      completedTime_s = submittedTime_s = lastAcceptedTime_s;
      encodedSteps = lastAcceptedSteps;
      if ((encodedSteps ?? 0) % 10 === 0 || performance.now() - lastProgressAt >= 1_000) {
        lastProgressAt = performance.now();
        writeProgress("frame-complete", { completedTime_s, encodedSteps,
          limiterPasses: frameTransport.lowFluxLimiter.totalPasses,
          transportSubsteps: frameTransport.executedSubsteps });
        await writeFile(output, `${JSON.stringify(
          makeReceipt(false, "worker has not completed"), null, 2)}\n`);
      }
    }

    const stats = await solver.readStats();
    completedTime_s = stats.completedTime_s ?? null;
    submittedTime_s = stats.submittedTime_s ?? null;
    encodedSteps = stats.encodedSteps ?? null;
    lastAcceptedTime_s = completedTime_s ?? lastAcceptedTime_s;
    lastAcceptedSteps = encodedSteps ?? lastAcceptedSteps;
    const timeTolerance = 8 * 2 ** -23 * duration_s;
    requireCondition(completedTime_s !== null
      && Math.abs(completedTime_s - duration_s) <= timeTolerance,
    `completed time ${completedTime_s} differs from requested ${duration_s} s`);
    requireCondition(submittedTime_s !== null
      && Math.abs(submittedTime_s - duration_s) <= timeTolerance,
    `submitted time ${submittedTime_s} differs from requested ${duration_s} s`);

    acceptedVolume = await solver.readAcceptedGeometricVolumeQA();
    requireCondition(acceptedVolume.invalidCells === 0,
      "final accepted volume contains invalid cells");
    requireCondition(acceptedVolume.nonfiniteCells === 0,
      "final accepted volume contains nonfinite cells");
    requireCondition(acceptedVolume.nonfiniteDynamicsCells === 0,
      "final accepted cells contain nonfinite dynamics");
    for (const name of ["volumeFine3", "volume_m3", "capacityFine3",
      "maximumBoundErrorFine3", "maximumBoundErrorPerFineCell"] as const) {
      requireCondition(Number.isFinite(acceptedVolume[name]),
        `accepted volume receipt has nonfinite ${name}`);
    }
    finiteFields = inspectFields(await solver.readDiagnosticFields(true));
    requireCondition(finiteFields.passed, `nonfinite diagnostic fields: ${JSON.stringify(
      finiteFields.nonfiniteByField)}`);
    const sourceEmittedFine3 = Number(transport?.hoseSourceLedger.emitted ?? 0);
    const initialVolumeFine3 = Number(initialAcceptedVolume.volumeFine3);
    const finalVolumeFine3 = Number(acceptedVolume.volumeFine3);
    const errorFine3 = finalVolumeFine3 - initialVolumeFine3
      - sourceEmittedFine3 + cumulativeOutflowFine3;
    const relativeError = Math.abs(errorFine3)
      / Math.max(Math.abs(initialVolumeFine3), Math.abs(sourceEmittedFine3), 1);
    const finalBalance: BalanceReceipt = { initialVolumeFine3, finalVolumeFine3, sourceEmittedFine3,
      cumulativeOutflowFine3, cumulativeOutflowQuantizationBoundFine3,
      errorFine3, relativeError, maximumRelativeError: BALANCE_RELATIVE_TOLERANCE,
      passed: relativeError < BALANCE_RELATIVE_TOLERANCE };
    balance = finalBalance;
    requireCondition(finalBalance.passed,
      `source/outflow-adjusted relative balance error ${relativeError} exceeds ${
        BALANCE_RELATIVE_TOLERANCE}`);
  } catch (error) {
    failure = error instanceof Error ? error.stack ?? error.message : String(error);
    const owner = error instanceof SimulationFailureError ? error.failure.ownerId : -1;
    if (solver) {
      // A rejected frame must not reset the receipt clock to the solver's
      // health-gated completedTime_s placeholder. These locals advance only
      // after awaitFrameCompletion accepted a frame above.
      completedTime_s = submittedTime_s = lastAcceptedTime_s;
      encodedSteps = lastAcceptedSteps;
      try {
        failureTransport = await solver.readGeometricVolumeTransportReceiptQA();
        rejectedFrame = {
          accepted: false,
          priorAcceptedTime_s: lastAcceptedTime_s,
          attemptedCompletedTime_s: Number.isFinite(failureTransport.physicalDt_s)
            ? lastAcceptedTime_s + failureTransport.physicalDt_s : null,
          physicalDt_s: failureTransport.physicalDt_s,
          executedPhysicalDt_s: failureTransport.executedPhysicalDt_s,
          plannedSubsteps: failureTransport.plannedSubsteps,
          executedSubsteps: failureTransport.executedSubsteps,
          fault: failureTransport.fault,
        };
      }
      catch { /* retain root failure */ }
      try { acceptedVolume ??= await solver.readAcceptedGeometricVolumeQA(); } catch { /* retain root failure */ }
      if (owner >= 0) {
        if (error instanceof SimulationFailureError
          && error.failure.kernel === "compileGeometricVolumeSubfaces") {
          try { failureOwnerRow = await solver.readAcceptedGeometricRowQA(owner); }
          catch { /* owner may belong to a rejected candidate */ }
        } else {
          try { failureOwnerCellRows = await solver.readAcceptedGeometricCellRowsQA(owner); }
          catch { /* owner may belong to a rejected candidate */ }
        }
      }
      try {
        const fields = await solver.readDiagnosticFields(true);
        finiteFields ??= inspectFields(fields);
      } catch { /* retain root failure */ }
    }
  } finally {
    clearInterval(initializationHeartbeat);
    try {
      sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(ROOT);
      if (sourceFingerprintAfter.sha256 !== sourceFingerprint.sha256 && failure === null) {
        failure = "Sparse Geometric sources changed during the scene run";
      }
    } catch (error) {
      if (failure === null) failure = `Could not fingerprint final sources: ${
        error instanceof Error ? error.message : String(error)}`;
    }
    const receipt = makeReceipt(failure === null, failure);
    try { await writeFile(output, `${JSON.stringify(receipt, null, 2)}\n`); }
    finally {
      if (device) {
        const manager = gpuCompilationManagerFor(device);
        try {
          await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
          solver?.destroy(); solver = undefined;
          invalidateGPUCompilationManager(device, "Sparse Geometric ladder scene complete");
          await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
        } catch { /* The scene receipt already carries the primary failure. */ }
        finally {
          solver?.destroy(); device.destroy();
          await new Promise<void>(resolve => setImmediate(resolve));
        }
      }
      await releaseWebGPUExclusiveLock();
    }
    if (failure) process.exitCode = 1;
  }
}

interface ChildResult {
  readonly code: number | null;
  readonly signal: NodeJS.Signals | null;
  readonly timedOut: boolean;
  readonly stderr: string;
}

async function removeDeadChildLock(pid: number | undefined): Promise<void> {
  const holder = await readWebGPUExclusiveLockHolder();
  if (pid !== undefined && holder?.owner?.pid === pid && !holder.alive) {
    await releaseWebGPUExclusiveLock();
  }
}

async function runScene(id: SparseCM12ComplexitySceneId, duration_s: number,
  timeoutMs: number, receiptPath: string): Promise<ChildResult> {
  const child = spawn(process.execPath, ["--import", "tsx", SELF], {
    cwd: ROOT,
    env: { ...process.env,
      WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE,
      FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
      FLUID_CM12_LADDER_CORRECTNESS_WORKER: "1",
      FLUID_CM12_LADDER_CORRECTNESS_SCENE: id,
      FLUID_CM12_LADDER_CORRECTNESS_DURATION_S: String(duration_s),
      FLUID_CM12_LADDER_CORRECTNESS_RECEIPT: receiptPath,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "", timedOut = false;
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => process.stderr.write(chunk));
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => { stderr += chunk; process.stderr.write(chunk); });
  const timer = setTimeout(() => {
    timedOut = true; child.kill("SIGTERM");
    setTimeout(() => child.kill("SIGKILL"), 2_000).unref();
  }, timeoutMs);
  try {
    const result = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>(
      (resolve, reject) => {
        child.once("error", reject);
        child.once("exit", (code, signal) => resolve({ code, signal }));
      });
    return { ...result, timedOut, stderr };
  } finally {
    clearTimeout(timer);
    await removeDeadChildLock(child.pid);
  }
}

async function supervisor(): Promise<void> {
  const requested = argument("scene") ?? "all";
  const selected = requested === "all" ? [...SPARSE_CM12_COMPLEXITY_SCENES]
    : [sceneDefinition(requested)];
  const duration_s = finiteNumber("duration", DEFAULT_DURATION_S);
  const timeoutMs = finiteNumber("timeout-ms", DEFAULT_SCENE_TIMEOUT_MS);
  const methodValues = resolveMethodValues(adaptiveMassMethod,
    SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE.quality,
    SPARSE_CM12_COMPLEXITY_LADDER_METHOD_PROFILE.overrides ?? {});
  if (flag("list")) {
    console.log(JSON.stringify({ duration_s, methodValues, scenes: selected }, null, 2)); return;
  }
  let holder = await readWebGPUExclusiveLockHolder();
  if (holder && !holder.alive) {
    await releaseWebGPUExclusiveLock();
    holder = await readWebGPUExclusiveLockHolder();
  }
  if (holder) throw new Error(`Cannot start ladder while ${holder.description} holds the WebGPU lease`);
  const sourceFingerprint = await fingerprintSparseCM12RepositorySources(ROOT);
  const temporary = await mkdtemp(join(tmpdir(), "fluid-geometric-ladder-"));
  const receipts: SceneReceipt[] = [];
  const startedAt = new Date().toISOString();
  const started = performance.now();
  const outputPath = argument("out");
  try {
    for (const definition of selected) {
      process.stderr.write(`[geometric-ladder] ${definition.id}: ${duration_s} s\n`);
      const receiptPath = join(temporary, `${definition.ordinal}-${definition.id}.json`);
      const child = await runScene(definition.id, duration_s, timeoutMs, receiptPath);
      let receipt: SceneReceipt;
      try {
        receipt = JSON.parse(await readFile(receiptPath, "utf8")) as SceneReceipt;
      } catch {
        receipt = { scene: definition.id, title: definition.title,
          methodValues, passed: false,
          durationRequested_s: duration_s, completedTime_s: null, submittedTime_s: null,
          encodedSteps: null, initialAcceptedVolume: null, acceptedVolume: null,
          finiteFields: null, balance: null, transport: null,
          failureTransport: null, rejectedFrame: null, validationErrors: [],
          deviceLost: null, wall_ms: 0,
          sourceFingerprint, sourceFingerprintAfter: null, sourceUnchanged: null,
          failure: child.timedOut ? `scene exceeded ${timeoutMs} ms timeout`
            : `worker exited ${child.code ?? child.signal ?? "without status"}: ${child.stderr.trim()}` };
      }
      if (child.timedOut || child.code !== 0) {
        const childFailure = child.timedOut
          ? `scene exceeded ${timeoutMs} ms timeout`
          : `worker exited ${child.code ?? child.signal ?? "without status"}`;
        const childDetail = `${childFailure}${child.stderr.trim() ? `: ${child.stderr.trim()}` : ""}`;
        const authoredFailure = receipt.failure?.trim();
        receipt = { ...receipt, passed: false,
          failure: !authoredFailure || authoredFailure === "worker has not completed"
            ? childDetail : `${authoredFailure}; ${childDetail}` };
      }
      receipts.push(receipt);
      const sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(ROOT);
      const partial = { suite: "sparse-geometric-ladder-3s", startedAt,
        method: "adaptive-volume", quality: "balanced", productionDefaults: true,
        methodValues,
        backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal", durationPerScene_s: duration_s,
        sourceFingerprint, sourceFingerprintAfter,
        sourceUnchanged: sourceFingerprintAfter.sha256 === sourceFingerprint.sha256,
        elapsedMs: Number((performance.now() - started).toFixed(1)),
        passed: sourceFingerprintAfter.sha256 === sourceFingerprint.sha256
          && receipts.length === selected.length && receipts.every(item => item.passed),
        scenes: receipts };
      if (outputPath) await writeFile(outputPath, `${JSON.stringify(partial, null, 2)}\n`);
    }
  } finally { await rm(temporary, { recursive: true, force: true }); }
  const sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(ROOT);
  const report = { suite: "sparse-geometric-ladder-3s", startedAt,
    method: "adaptive-volume", quality: "balanced", productionDefaults: true,
    methodValues,
    backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal", durationPerScene_s: duration_s,
    sourceFingerprint, sourceFingerprintAfter,
    sourceUnchanged: sourceFingerprintAfter.sha256 === sourceFingerprint.sha256,
    elapsedMs: Number((performance.now() - started).toFixed(1)),
    passed: sourceFingerprintAfter.sha256 === sourceFingerprint.sha256
      && receipts.length === selected.length && receipts.every(item => item.passed),
    scenes: receipts };
  if (outputPath) await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) process.exitCode = 1;
}

if (process.env.FLUID_CM12_LADDER_CORRECTNESS_WORKER === "1") await worker();
else await supervisor();
