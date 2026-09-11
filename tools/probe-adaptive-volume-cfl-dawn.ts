/** Production Sparse Geometric large-outer-step probe; no browser or unit runner.
 * Run serially under the repository Dawn lease. --help does not acquire a GPU.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { createMinimalPowerDamBreakScene } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { fluidExecutionDeviceFeatures } from "../lib/core/gpu-startup";
import { GPUStageTimestampRecorder } from "../lib/core/performance-trace";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { gpuCompilationManagerFor, invalidateGPUCompilationManager } from "../lib/core/gpu-compilation-manager";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod as adaptiveVolumeMethod } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { fingerprintSparseCM12RepositorySources } from "./sparse-cm12-source-content-fingerprint";

const arg = (name: string, fallback: string) => process.argv.slice(2)
  .find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (process.argv.includes("--help")) {
  console.log(`Sparse Geometric production CFL probe
node --import tsx tools/probe-adaptive-volume-cfl-dawn.ts [options]
  --cases=1,2,4,8,16,25  Initial finest-cell CFL targets (default shown)
  --initial-speed=4      Initial +X speed in metres/second; gravity stays authored
  --duration=0.5         Matched simulated duration for every case, seconds
  --dt=SECONDS          Run one explicit outer-dt case instead of --cases
  --steps=N             Explicit-dt mode only; duration becomes N*dt
  --inflow=none|submerged|above  Add a small continuous production hose
  --out=PATH            Optional complete JSON receipt
Final remainder steps are shorter to match duration. Requested CFL is based on
initial speed; actual evolving transport Courant and substeps are GPU receipts.
Uses the existing mini16 production scene and balanced method settings with only
scene timestep selection overridden. Bounds use production 8*FLT_EPS*C; mass
limit is the unchanged 0.005 relative mini32 Dawn conservation gate. No surface
isosurface or rendered occupancy is substituted for accepted physical volume.`);
  process.exit(0);
}
const inflowMode = arg("inflow", "none");
assert.ok(["none", "submerged", "above"].includes(inflowMode), "unknown inflow mode");
const speed = Number(arg("initial-speed", "4"));
const explicitDt = arg("dt", "");
const stepsOption = arg("steps", "");
let duration = Number(arg("duration", "0.5"));
assert.ok(Number.isFinite(speed) && speed > 0, "initial-speed must be finite and positive");
if (stepsOption) {
  assert.ok(explicitDt, "--steps requires --dt");
  const steps = Number(stepsOption);
  assert.ok(Number.isSafeInteger(steps) && steps > 0, "steps must be a positive integer");
  duration = Number(explicitDt) * steps;
}
assert.ok(Number.isFinite(duration) && duration > 0, "duration must be finite and positive");
const finestCell_m = 0.05; // Existing mini16 authored voxel domain, checked below.
const cases = explicitDt ? [{ targetCFL: Number(explicitDt) * speed / finestCell_m, dt_s: Number(explicitDt) }]
  : arg("cases", "1,2,4,8,16,25").split(",").map(value => ({
    targetCFL: Number(value), dt_s: Number(value) * finestCell_m / speed,
  }));
assert.ok(cases.length > 0 && cases.every(c => Number.isFinite(c.dt_s) && c.dt_s > 0), "CFL/dt must be finite and positive");
const root = fileURLToPath(new URL("..", import.meta.url));
const identity = await fingerprintSparseCM12RepositorySources(root);
const values = resolveMethodValues(adaptiveVolumeMethod, "balanced", { timeStep: "scene" });
assert.equal(adaptiveVolumeMethod.id, "adaptive-volume");
const report: { [key: string]: unknown; cases: unknown[] } = {
  probe: "adaptive-volume-cfl-dawn", methodId: adaptiveVolumeMethod.id,
  implementation: {
    solver: "lib/methods/adaptive-volume/webgpu-adaptive-mass-solver.ts",
    resident: "lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts",
    volumeShader: "lib/methods/adaptive-volume/resident-volume.wgsl.ts",
    sourceFingerprint: identity,
    commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
  },
  configuration: { duration_s: duration, initialSpeed_m_s: speed, finestCell_m, inflowMode, methodValues: values,
    conservationRelativeLimit: 0.005, conservationLimitSource: "tests/sparse-cm12-mini32-corner-dawn.test.ts",
    timestampQuantum_us: 65.536, diagnosticReadbacksExcludedFromWallTime: true },
  cases: [],
};
const pause = (ms: number) => new Promise<void>(done => setTimeout(done, ms));
let anyFailure = false;
await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-adaptive-volume-cfl-dawn.ts");
try {
  usePerformanceInstrumentationStore.getState().setMode("timeline");
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(resolve(modulePath)).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  for (const configuration of cases) {
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    const frames: unknown[] = [];
    const entry: Record<string, unknown> = { ...configuration, frames, passed: false };
    report.cases.push(entry);
    const validationErrors: string[] = [];
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn adapter unavailable");
      assert.ok(adapter.features.has("timestamp-query"), "hardware timestamps required");
      device = await adapter.requestDevice({ requiredFeatures: fluidExecutionDeviceFeatures(adapter.features),
        requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      device.addEventListener("uncapturederror", event => validationErrors.push(event.error.message));
      await GPUStageTimestampRecorder.prepare(device);
      const scene = createMinimalPowerDamBreakScene();
      assert.equal(scene.voxelDomain?.finestCellSize_m, finestCell_m);
      scene.duration_s = duration;
      scene.numerics.fixedDt_s = scene.numerics.maxDt_s = configuration.dt_s;
      scene.fluid.initialVelocity_m_s = { x: speed, y: 0, z: 0 };
      if (inflowMode !== "none") {
        scene.fluid.inflow = {
          center_m: { x: -0.25, y: inflowMode === "submerged" ? 0.15 : 0.65, z: 0 },
          radius_m: 0.035, length_m: 0.05,
          velocity_m_s: { x: 1, y: 0, z: 0 }, start_s: 0, end_s: duration + 1, ramp_s: 0,
        };
      }
      const start = performance.now();
      solver = await adaptiveVolumeMethod.createSolverAsync!(device, scene, "balanced", values,
        undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      entry.construction_ms = performance.now() - start;
      const initial = await solver.readAcceptedGeometricVolumeQA();
      entry.initial = initial;
      assert.equal(initial.invalidCells, 0, "initial physical volume bounds");
      let t = 0, outflowFine3 = 0, previousSample = -1;
      while (t < duration - 1e-10) {
        const target = Math.min(duration, t + configuration.dt_s);
        const requestedDt = target - t;
        await pause(110); // Recorder capture cadence, outside measured work.
        const started = performance.now();
        const deadline = started + 120_000;
        while (!solver.advanceTo(target, [])) {
          assert.ok(performance.now() < deadline, "advance timed out awaiting accepted topology");
          await new Promise<void>(done => setImmediate(done));
        }
        await solver.awaitFrameCompletion();
        await device.queue.onSubmittedWorkDone();
        const wall_ms = performance.now() - started;
        const transport = await solver.readGeometricVolumeTransportReceiptQA();
        outflowFine3 += transport.outflowFineCells3;
        const physical = await solver.readAcceptedGeometricVolumeQA();
        const world = await solver.readWorldGrowthReceiptQA();
        const stats = await solver.readStats();
        let trace = solver.readPerformanceTraceSnapshot().physicsTrace;
        for (let poll = 0; poll < 30 && (!trace || trace.sampleId <= previousSample
          || trace.context !== `adaptive-volume:sim-${target.toFixed(6)}`); poll++) {
          await pause(5); trace = solver.readPerformanceTraceSnapshot().physicsTrace;
        }
        const traceFresh = !!trace && trace.sampleId > previousSample
          && trace.context === `adaptive-volume:sim-${target.toFixed(6)}`;
        if (traceFresh) previousSample = trace!.sampleId;
        const sourceEmitted = transport.hoseSourceLedger.emitted ?? 0;
        const balanceErrorFine3 = physical.volumeFine3 - initial.volumeFine3 - sourceEmitted + outflowFine3;
        const relativeBalanceError = Math.abs(balanceErrorFine3) / Math.max(initial.volumeFine3, 1e-30);
        frames.push({ targetTime_s: target, requestedDt_s: requestedDt, wall_ms, transport, physical,
          outflowCumulativeFine3: outflowFine3, sourceEmittedFine3: sourceEmitted,
          dynamicWorld: { liquidMassFine3: world.dynamicLiquidMassFineCells,
            liquidBoundsFine: world.dynamicLiquidBoundsFine },
          balanceErrorFine3, relativeBalanceError, completedTime_s: stats.completedTime_s,
          encodedSteps: stats.encodedSteps, topologyGenerationCount: stats.topologyGenerationCount,
          gpu: traceFresh ? { measurementSource: trace!.measurementSource, total_ms: trace!.total_ms,
            phases: trace!.phases.map(phase => ({ label: phase.label, duration_ms: phase.duration_ms })) } : null });
        assert.equal(transport.fault, 0, "transport fault");
        assert.ok(transport.transportCompleted, "transport microsteps did not complete");
        assert.ok(Math.abs(transport.executedPhysicalDt_s - requestedDt) <= 8 * 2 ** -23 * requestedDt,
          "actual executed duration differs from requested outer step");
        assert.equal(physical.invalidCells, 0, "accepted physical volume bounds");
        assert.ok(relativeBalanceError < 0.005, "source/outflow-adjusted conservation gate");
        assert.ok(traceFresh && trace?.measurementSource === "gpu-hardware-timestamp", "missing fresh hardware stage trace");
        assert.deepEqual(validationErrors, []);
        t = target;
      }
      if (inflowMode !== "none") {
        const source = (await solver.readGeometricVolumeTransportReceiptQA()).hoseSourceLedger;
        assert.ok(source.requested > 0, "the production hose must record requested volume");
        assert.ok(source.emitted > 0, "the production hose must emit into the supported fluid component");
        assert.equal(source.fault, 0, "hose volume ledger fault");
        assert.ok(Math.abs(source.requested - source.emitted - source.pending)
          <= 8 * 2 ** -23 * source.requested, "requested = emitted + pending hose ledger");
      }
      const fields = await solver.readDiagnosticFields();
      entry.finalFieldDigests = Object.fromEntries([
        "density", "velocity", "pressure", "divergence",
      ].map(name => {
        const field = fields[name as "density" | "velocity" | "pressure" | "divergence"];
        return [name, createHash("sha256").update(new Uint8Array(
          field.buffer, field.byteOffset, field.byteLength)).digest("hex")];
      }));
      entry.finalFieldDigestScope = "authored diagnostic lattice; outside-world amount is checked separately by accepted volume QA";
      entry.passed = true;
    } catch (error) {
      anyFailure = true; entry.failure = error instanceof Error ? error.stack : String(error);
      if (solver) {
        try { entry.failureTransport = await solver.readGeometricVolumeTransportReceiptQA(); } catch { /* original failure retained */ }
        try { entry.failureComponents = await solver.readGeometricProjectionComponentsQA(); } catch { /* original failure retained */ }
      }
    } finally {
      entry.validationErrors = validationErrors;
      if (device) {
        const manager = gpuCompilationManagerFor(device);
        try {
          await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
          solver?.destroy(); solver = undefined;
          invalidateGPUCompilationManager(device, "CFL case complete");
          await manager.whenIdle(); await device.queue.onSubmittedWorkDone();
        } finally {
          solver?.destroy(); device.destroy();
          await new Promise<void>(done => setImmediate(done));
        }
      }
    }
    console.error(JSON.stringify({ targetCFL: configuration.targetCFL, passed: entry.passed, frames: frames.length }));
  }
} catch (error) {
  anyFailure = true;
  report.failure = error instanceof Error ? error.stack : String(error);
} finally {
  await releaseWebGPUExclusiveLock();
  report.sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(root);
  report.sourceUnchanged = (report.sourceFingerprintAfter as { sha256: string }).sha256 === identity.sha256;
  report.passed = !anyFailure && report.sourceUnchanged;
  const output = JSON.stringify(report, null, 2);
  console.log(output);
  if (arg("out", "")) await writeFile(resolve(arg("out", "")), `${output}\n`);
}
assert.equal((report.sourceFingerprintAfter as { sha256: string }).sha256, identity.sha256, "source changed during run");
if (anyFailure) process.exitCode = 1;
