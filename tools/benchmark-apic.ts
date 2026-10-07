/** Serial, fenced solver throughput at equal physical endpoints. Not browser FPS.
 *   node --import tsx tools/benchmark-apic.ts --methods=apic-indirect,apic-direct,uniform-volume
 *   --scenes=rest16,drop16,dam32 --steps-ms=8.333333333,16.666666667,33.333333333
 *   --profile-gpu records diagnostic pass timestamps (not throughput); --progress logs clock intervals.
 *   --transfer-mode=apic|pic|flip --flip-ratio=0.95 selects particle transfer.
 *   --duration=0.2 --warmup=0.05 --out=docs/verification/apic-performance.json
 */
import "../lib/methods";
import { GPUPassProfile } from "./gpu-pass-profile";
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { getMethod } from "../lib/core/method-registry";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { APIC_CONFIGURATION } from "../lib/methods/particle/method";
import { MacGridSolver } from "../lib/methods/mac-shared/solver";
import type { ApicInfo } from "../lib/methods/particle/transport";
import { readFloatTexture3D } from "../lib/harness/webgpu-smoke-readbacks";

const arg = (key: string, fallback: string) => process.argv.find(v => v.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const scenes = arg("scenes", "rest16,drop16,dam32").split(",");
const methods = arg("methods", "apic-direct,apic-mg,apic-mg-large,uniform-full,uniform-volume").split(",");
const steps = arg("steps-ms", "8.333333333,16.666666667,33.333333333").split(",").map(v => Number(v) / 1000);
const transferMode = arg("transfer-mode", "apic");
const flipRatio = Number(arg("flip-ratio", "0.95"));
assert.ok(["apic", "pic", "flip"].includes(transferMode) && flipRatio >= 0 && flipRatio <= 1);
const pressureTolerance = Number(arg("pressure-tolerance", "5"));
const pressureRelativeReduction = Number(arg("pressure-relative", "0.1"));
assert.ok(pressureTolerance >= 0.00001 && pressureTolerance <= 100 && pressureRelativeReduction >= 0 && pressureRelativeReduction <= 1);
const duration = Number(arg("duration", "0.2")), warmup = Number(arg("warmup", "0.05"));
const mgIterations = Number(arg("mg-iterations", String(APIC_CONFIGURATION.resolveOptions!(
  sceneDocument(getSceneDefinition("minimal-power-dam-break-32")), {}).pressureLimit)));
const cfl = Number(arg("cfl", "0.5")), advanceCapacity = Number(arg("advance-capacity", "1"));
assert.ok(Number.isInteger(mgIterations) && mgIterations >= 1 && mgIterations <= 128);
assert.ok(cfl >= 0.1 && cfl <= 4 && advanceCapacity >= 1 && advanceCapacity <= 64);
assert.ok(Number.isFinite(duration) && Number.isFinite(warmup) && duration > 0 && warmup >= 0 && steps.every(dt => dt >= 0.0001 && dt <= 0.05));
assert.ok(scenes.every(s => ["rest16", "drop16", "dam32", "dam64"].includes(s)));
assert.ok(methods.every(s => ["apic-indirect", "apic-direct", "apic-batched", "apic-mg", "apic-mg-large", "uniform-full", "uniform-volume"].includes(s)));
const output = resolve(arg("out", "docs/verification/apic-performance.json"));
const percentile = (samples: number[], q: number) => [...samples].sort((a, b) => a - b)[Math.max(0, Math.ceil(samples.length * q) - 1)];
const sourcePaths = ["tools/benchmark-apic.ts", "tools/gpu-pass-profile.ts", "lib/methods/particle/method.ts", "lib/methods/particle/shader.ts", "lib/methods/particle/transport.ts", "lib/methods/particle/parameters.ts", "lib/methods/particle/scan.ts", "lib/methods/mac-shared/solver.ts", "lib/methods/mac-shared/shader.ts", "lib/methods/mac-shared/multigrid.ts", "lib/methods/mac-shared/schedule.ts", "lib/methods/mac-shared/pressure-target.ts", "lib/methods/mac-shared/transport.ts"];
const report = { date: new Date().toISOString(), node: process.version, revision: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
  sourceHashes: Object.fromEntries(sourcePaths.map(p => [p, createHash("sha256").update(readFileSync(p)).digest("hex")])),
  methodology: "Each requested clock step is completed and health-checked before the next. Wall times include CPU encoding, GPU execution, fence/receipt overhead; exclude construction and surface readback. No renderer. p95 is a short-run sample, not a 60 FPS acceptance result. All arms end at the same physical time. uniform-volume uses app defaults with the shared timestep; uniform-full overrides detailPolicy=full. Defaults have different pressure tolerances and are not matched fidelity. Native publication dimensions may differ; surface differences are relative to the first successful arm with matching dimensions at each scene/timestep, not ground truth.",
  duration_s: duration, warmup_s: warmup, adapter: {}, runs: [] as Record<string, unknown>[] };
const save = () => { mkdirSync(dirname(output), { recursive: true }); writeFileSync(output, JSON.stringify(report, null, 2) + "\n"); };

let announcedWait = false;
for (;;) {
  try { await acquireWebGPUExclusiveLock("dawn-probe", "APIC timestep and dispatch performance comparison"); break; }
  catch (error) {
    if (!String(error).includes("Refusing concurrent GPU execution")) throw error;
    if (!announcedWait) { console.log("Waiting for the repository WebGPU lease."); announcedWait = true; }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
}
const profileEnabled = process.argv.includes("--profile-gpu");
let passProfile: GPUPassProfile | undefined;
let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  report.adapter = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description };
  const raw = await adapter.requestDevice({ requiredFeatures: profileEnabled ? ["timestamp-query"] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  if (profileEnabled) passProfile = new GPUPassProfile(raw);
  device = managedGPUDevice(passProfile?.device ?? raw, { requireWorkerRealm: false });
  const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  for (const name of scenes) for (const dt of steps) {
    const scene = structuredClone(sceneDocument(getSceneDefinition(name === "dam64" ? "minimal-power-dam-break-64" : "minimal-power-dam-break-32")));
    scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };
    if (name === "rest16" || name === "drop16") {
      Object.assign(scene.container, { width_m: 1, height_m: 1, depth_m: 1, top: "closed", fluidWallMode: "free-slip", fillFraction: name === "rest16" ? 0.5 : 0 });
      scene.voxelDomain.finestCellSize_m = 1 / 16; scene.rigidBodies = []; scene.solidVoxels = [];
      Object.assign(scene.fluid, { initialCondition: "tank-fill", initialBrickSeeds_m: undefined, initialHeightField: undefined, initialVelocity_m_s: undefined,
        initialDamBreakDimensions_m: undefined, initialDamBreakOrigin_m: undefined,
        inflow: undefined, surfaceTension_N_m: 0, dynamicViscosity_Pa_s: 0, gravity_m_s2: { x: 0, y: -9.81, z: 0 },
        initialLiquidVolumes: name === "rest16" ? [] : [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.18 }] });
    }
    let reference: Float32Array | undefined;
    for (const variant of methods) {
      let solver: GPUSolverInstance | undefined;
      const run: Record<string, unknown> = { scene: name, variant, requestedDt_ms: dt * 1000,
        sceneHash: createHash("sha256").update(JSON.stringify(scene)).digest("hex"), sceneDocument: scene };
      try {
        const init = performance.now();
        if (variant.startsWith("apic-")) {
          const multigridPressure = variant.startsWith("apic-mg"), large = variant === "apic-mg-large";
          const runCfl = large ? APIC_CONFIGURATION.resolveOptions!(scene, {}).cfl : cfl, runCapacity = large ? APIC_CONFIGURATION.advanceStepCapacity! : variant === "apic-batched" ? 4 : advanceCapacity;
          run.pressureTolerance = pressureTolerance; run.pressureRelativeReduction = pressureRelativeReduction;
          run.transferMode = transferMode; run.flipRatio = flipRatio;
          run.pressureIterationLimit = multigridPressure ? mgIterations : 256;
          run.cfl = runCfl; run.advanceCapacity = runCapacity;
          run.substepCapacityFactor = large ? APIC_CONFIGURATION.substepCapacityFactor : 1;
          run.commandBufferSubsteps = APIC_CONFIGURATION.commandBufferSubsteps;
          run.continuationSubsteps = APIC_CONFIGURATION.continuationSubsteps;
          run.advanceCellStepBudget = APIC_CONFIGURATION.advanceCellStepBudget;
          solver = await MacGridSolver.createAsync(device, structuredClone(scene), "balanced", { transferMode, flipRatio, pressureTolerance, pressureRelativeReduction }, undefined, undefined,
            { ...APIC_CONFIGURATION, directPressure: variant !== "apic-indirect", advanceStepCapacity: runCapacity, multigridPressure, substepCapacityFactor: large ? APIC_CONFIGURATION.substepCapacityFactor : 1,
              resolveOptions: (s, v) => ({ ...APIC_CONFIGURATION.resolveOptions!(s, v), cfl: runCfl, pressureLimit: multigridPressure ? mgIterations : 256 }) });
        }
        else {
          const method = getMethod("uniform-volume"), values = resolveMethodValues(method, "balanced", { ...method.appDefaults, timeStep: "scene", ...(variant === "uniform-full" ? { detailPolicy: "full" } : {}) });
          run.methodValues = values;
          solver = await method.createSolverAsync!(device, structuredClone(scene), "balanced", values, undefined, () => {});
        }
        const initial = await solver.readStats(); run.initialization_ms = performance.now() - init;
        run.initialVolumeCellSum = initial.volumeCellSum;
        assert.ok((initial.volumeCellSum ?? 0) > 0, "fixture must initialize nonempty liquid");
        const profileRows: Record<string, unknown>[] = [];
        const clockRows: Record<string, unknown>[] = [];
        const clockStep = async (target: number) => {
          let encode = 0, advances = 0, pressureIterations = 0;
          while ((solver!.info.submittedTime_s ?? 0) < target - 1e-9) {
            assert.ok(++advances < 1000, "clock step failed to finish");
            if (passProfile && target > warmup + 1e-9) passProfile.start();
            const start = performance.now(); solver!.advanceTo(target, []); const synchronousEncode = performance.now() - start;
            if (solver!.awaitFrameCompletion) await solver!.awaitFrameCompletion(); else await solver!.readStats();
            if (variant.startsWith("apic-")) {
              const info = solver!.info;
              const expected = pressureRelativeReduction > 0 ? Math.min(pressureTolerance, Math.max(Math.fround(pressureRelativeReduction) * info.pressureInitialResidual!, 0.0001)) : pressureTolerance;
              assert.ok(Math.abs(info.pressureResidualTarget! - expected) <= 1e-6 * Math.max(1, expected), "pressure target must survive particle-ledger publication");
              assert.ok(info.pressureResidual! <= info.pressureResidualTarget!, "accepted residual exceeds the effective target");
            }
            encode += (solver!.info as ApicInfo).macFrameEncode_ms ?? synchronousEncode;
            if (passProfile && target > warmup + 1e-9) profileRows.push({ target_s: target,
              completedTime_s: solver!.info.completedTime_s, substeps: solver!.info.lastSubsteps,
              pressureIterations: (solver!.info as ApicInfo).macFramePressureIterations, encodedSlots: (solver!.info as ApicInfo).macFrameSlots,
              passes: await passProfile.finish() });
            pressureIterations += (solver!.info as ApicInfo).macFramePressureIterations ?? solver!.info.pressureIterationsExecuted ?? 0;
          }
          return { encode, advances, pressureIterations };
        };
        for (let time = 0; time < warmup - 1e-9;) { time = Math.min(warmup, time + dt); await clockStep(time); }
        const warmupSubsteps = solver.info.encodedSteps ?? 0;
        const wall: number[] = [], cpu: number[] = []; let advances = 0, pressureIterations = 0;
        for (let time = warmup; time < warmup + duration - 1e-9;) {
          time = Math.min(warmup + duration, time + dt);
          const start = performance.now(), result = await clockStep(time);
          wall.push(performance.now() - start); cpu.push(result.encode);
          clockRows.push({ target_s: time, wall_ms: wall.at(-1), cpuEncode_ms: result.encode, advances: result.advances,
            pressureResidual: solver.info.pressureResidual, pressureResidualTarget: solver.info.pressureResidualTarget, divergence_s: solver.info.maxDivergenceAfter_s,
            pressureIterations: result.pressureIterations, lifetimeSubsteps: solver.info.encodedSteps,
            maxSpeed_m_s: solver.info.maxSpeed_m_s, affineSpeedBound_m_s: (solver.info as ApicInfo).apicAffineSpeedBound_m_s });
          if (process.argv.includes("--progress")) console.log(JSON.stringify({ variant, ...clockRows.at(-1) })); advances += result.advances; pressureIterations += result.pressureIterations;
        }
        run.clockRows = clockRows;
        if (passProfile) { run.profileRows = profileRows; run.timingCaveat = "Instrumented pass timestamps; wall includes profile readback. Use uninstrumented arms for throughput."; }
        const info = await solver.readStats() as ApicInfo;
        const total = wall.reduce((a, b) => a + b, 0);
        Object.assign(run, { ok: true, completedTime_s: info.completedTime_s ?? info.simulatedTime_s,
          clockSteps: wall.length, advances, gpuSubsteps: info.encodedSteps, wallTotal_ms: total, realtimeFactor: duration * 1000 / total,
          measuredGpuSubsteps: (info.encodedSteps ?? 0) - warmupSubsteps, pressureIterations,
          averageSubstep_ms: info.encodedSteps ? duration * 1000 / (info.encodedSteps - warmupSubsteps) : undefined,
          wallP50_ms: percentile(wall, 0.5), wallP95_ms: percentile(wall, 0.95), cpuEncodeP50_ms: percentile(cpu, 0.5),
          allocatedBytes: info.allocatedBytes, particles: info.apicParticleCount, lastDt_ms: (info.lastDt_s ?? 0) * 1000,
          maxSpeed_m_s: info.maxSpeed_m_s, divergence_s: info.maxDivergenceAfter_s, pressureResidual: info.pressureResidual, pressureConverged: info.pressureSolveConverged, pressureInitialResidual: info.pressureInitialResidual, pressureResidualTarget: info.pressureResidualTarget,
          finalPressureIterations: info.pressureIterationsExecuted,
          surfaceVolumeDrift: info.volumeDrift, finalVolumeCellSum: info.volumeCellSum, materialBalanceError: info.apicMaterialDrift, particleEnergy_J: info.apicKineticEnergy_J,
          uniformFineTiles: info.uniformMixedFineTiles, uniformCoarseTiles: info.uniformMixedCoarseTiles,
          uniformAcceptedResidual: info.uniformPressureAcceptedResidual,
          dimensions: [info.nx, info.ny, info.nz], effectiveSolveCells: info.cellCount });
        if (solver.denseLevelSetVolumeSource) {
          const texture = solver.denseLevelSetVolumeSource.vertexPhi;
          run.publishedSurfaceDimensions = [texture.width, texture.height, texture.depthOrArrayLayers];
          const phi = await readFloatTexture3D(device, texture, texture.width, texture.height, texture.depthOrArrayLayers);
          assert.ok(phi.every(Number.isFinite));
          if (reference?.length === phi.length) {
            let squares = 0, maximum = 0, signs = 0;
            for (let i = 0; i < phi.length; i++) { const delta = Math.abs(phi[i] - reference[i]); squares += delta * delta; maximum = Math.max(maximum, delta); signs += Number((phi[i] < 0) !== (reference[i] < 0)); }
            Object.assign(run, { phiRmsDifference_m: Math.sqrt(squares / phi.length), phiMaxDifference_m: maximum, signDisagreementFraction: signs / phi.length });
          } else reference = phi;
        }
        assert.deepEqual(errors, []);
      } catch (error) { Object.assign(run, { ok: false, error: String(error), gpuErrors: [...errors],
        completedTime_s: solver?.info.completedTime_s, submittedTime_s: solver?.info.submittedTime_s,
        pressureResidual: solver?.info.pressureResidual, finalPressureIterations: solver?.info.pressureIterationsExecuted }); }
      finally { solver?.destroy(); await device.queue.onSubmittedWorkDone(); }
      report.runs.push(run); save(); console.log(JSON.stringify({ ...run, sceneDocument: undefined, clockRows: undefined, profileRows: undefined }));
    }
  }
} finally { passProfile?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); save(); }
if (report.runs.some(run => !run.ok)) process.exitCode = 1;
