import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import {
  powerDamResultFromLine,
  powerDamResultWindow,
  powerDamPerformanceFailures,
  summarizePowerDamPerformance,
  type PowerDamResultRecord,
} from "./power-dam-performance-report";
import {
  buildOctreeRegressionArtifact,
  type OctreeRegressionLane,
  type OctreeRegressionResultRecord,
} from "./octree-regression-artifact";

type RuntimeLane = "mini" | "ui" | "moving-interface";

const args = new Set(process.argv.slice(2));
const requestedLane = process.argv.find((argument) => argument.startsWith("--lane="))
  ?.slice("--lane=".length) ?? "mini";
const quiescent = args.has("--quiescent") || requestedLane === "quiescent";
if (requestedLane !== "mini" && requestedLane !== "ui" && requestedLane !== "quiescent"
  && requestedLane !== "moving-interface") {
  throw new Error("--lane must be mini, ui, quiescent, or moving-interface");
}
if (quiescent && requestedLane === "ui") {
  throw new Error("--quiescent uses the mini dam and cannot be combined with --lane=ui");
}
const lane = (requestedLane === "quiescent" ? "moving-interface" : requestedLane) as RuntimeLane;
const traceProfile = args.has("--profile");
const fineTimestamps = args.has("--fine-timestamps");
if (quiescent && fineTimestamps) {
  throw new Error("--quiescent cannot use --fine-timestamps until the runner can timestamp only a trailing window");
}
const jsonOnly = args.has("--json");
const artifactPath = process.argv.find((argument) => argument.startsWith("--artifact="))
  ?.slice("--artifact=".length);
const root = fileURLToPath(new URL("..", import.meta.url));
const runner = fileURLToPath(new URL("./run-webgpu-smoke-isolated.ts", import.meta.url));

const laneEnvironment: Record<RuntimeLane, Record<string, string>> = {
  mini: {
    FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "2",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "500", FLUID_EXPECT_EXACT_STEPS: "500",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "2",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  "moving-interface": {
    FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "0.248",
    FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "16,16,16", FLUID_MAXIMUM_LEAF_SIZE: "2",
    FLUID_OCTREE_INTERFACE_BAND: "3", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
  },
  ui: {
    FLUID_SCENE: "dam-break-ui", FLUID_TARGET_S: "0.496",
    FLUID_MAX_DT: "0.008", FLUID_ORACLE_STEPS: "62", FLUID_EXPECT_EXACT_STEPS: "62",
    FLUID_EXPECT_GRID: "24,18,16",
  },
};

const benchmarkEnvironment = (overrides: Record<string, string> = {}): NodeJS.ProcessEnv => ({
    ...process.env,
    WEBGPU_NODE_MODULE: `${root}/node_modules/webgpu/index.js`,
    FLUID_WEBGPU_BACKEND: "metal",
    FLUID_WEBGPU_ADAPTER: "Apple M1 Max",
    // Dawn's unchecked Metal path can abort inside native code instead of
    // reporting a bad command. Keep the throughput default, but permit an
    // evidence-grade validated run when diagnosing or comparing a cut.
    FLUID_WEBGPU_DAWN_FEATURES: process.env.FLUID_BENCHMARK_VALIDATE === "1"
      ? "" : "skip_validation",
    FLUID_METHOD: "octree",
    FLUID_QUALITY: "balanced",
    FLUID_PERFORMANCE_PROFILE: "1",
    FLUID_PERFORMANCE_TRACES: traceProfile || artifactPath ? "1" : "0",
    FLUID_GPU_FINE_TIMESTAMPS: fineTimestamps ? "1" : "0",
    FLUID_GPU_COMMAND_AUDIT: "1",
    FLUID_STABILITY_ENVELOPE: artifactPath ? "1" : (process.env.FLUID_STABILITY_ENVELOPE ?? "0"),
    FLUID_REGRESSION_ARTIFACT: artifactPath ? "1" : "0",
    FLUID_CHECKPOINT_EVERY_S: artifactPath
      ? (lane === "ui" ? "0.08" : lane === "mini" ? "0.1" : "0.04")
      : (process.env.FLUID_CHECKPOINT_EVERY_S ?? "0"),
    FLUID_CPU_ORACLE: "0", FLUID_FIELD_STATS: "0", FLUID_SPARSE_STATS: "0",
    FLUID_RASTER_CHECKPOINTS: "0", FLUID_WEBGPU_SMOKE_TIMEOUT_MS: "240000",
    ...laneEnvironment[lane],
    ...overrides,
});

const runBenchmark = async (overrides: Record<string, string> = {}): Promise<PowerDamResultRecord> => {
  const child = spawn(process.execPath, ["--import", "tsx", runner], {
    cwd: root,
    env: benchmarkEnvironment(overrides),
    stdio: ["ignore", "pipe", "inherit"],
  });
  let result: PowerDamResultRecord | undefined;
  const lines = createInterface({ input: child.stdout! });
  lines.on("line", (line) => { result = powerDamResultFromLine(line) ?? result; });
  const exitCode = await new Promise<number>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => signal
      ? reject(new Error(`Dawn benchmark exited from ${signal}`))
      : resolve(code ?? 1));
  });
  if (exitCode !== 0) throw new Error(`Dawn benchmark exited with code ${exitCode}`);
  if (!result) throw new Error("Dawn benchmark completed without a result record");
  return result;
};

const movingResult = await runBenchmark();
let artifactResult: OctreeRegressionResultRecord = movingResult;
let summary = summarizePowerDamPerformance(movingResult);
if (quiescent) {
  const settleSteps = 500;
  const measuredSteps = 60;
  const environmentForSteps = (steps: number): Record<string, string> => ({
    FLUID_TARGET_S: String(steps * 0.004),
    FLUID_ORACLE_STEPS: String(steps),
    FLUID_EXPECT_EXACT_STEPS: String(steps),
  });
  const settlePrefix = await runBenchmark(environmentForSteps(settleSteps));
  const complete = await runBenchmark(environmentForSteps(settleSteps + measuredSteps));
  artifactResult = powerDamResultWindow(settlePrefix, complete);
  const quiescentSummary = summarizePowerDamPerformance(artifactResult);
  if (!jsonOnly) {
    console.log(`${summary.scenario}: ${summary.advanceWall_ms.toFixed(2)} ms/advance (${summary.steps} advances, in motion)`);
  }
  summary = quiescentSummary;
}
if (artifactPath) {
  const artifact = buildOctreeRegressionArtifact({
    lane: requestedLane as OctreeRegressionLane,
    result: artifactResult,
    repositoryRoot: root,
    adapter: process.env.FLUID_WEBGPU_ADAPTER ?? "Apple M1 Max",
  });
  const absoluteArtifactPath = resolve(root, artifactPath);
  mkdirSync(dirname(absoluteArtifactPath), { recursive: true });
  writeFileSync(absoluteArtifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  if (!jsonOnly) console.log(`regression artifact: ${absoluteArtifactPath}`);
  if (artifact.blockers.length > 0) {
    for (const entry of artifact.blockers) {
      console.error(`regression artifact blocker: ${entry.metric}: ${entry.reason}`);
    }
    process.exitCode = 1;
  }
}
const numericLimit = (name: string): number | undefined => {
  const raw = process.env[name];
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive and finite`);
  return value;
};
const failures = powerDamPerformanceFailures(summary, {
  maximumAdvanceWall_ms: numericLimit("FLUID_MAX_ADVANCE_MS"),
  maximumDispatchesPerAdvance: numericLimit("FLUID_MAX_DISPATCHES_PER_ADVANCE"),
  maximumComputePassesPerAdvance: numericLimit("FLUID_MAX_PASSES_PER_ADVANCE"),
  maximumPressureNonSolve_ms: numericLimit("FLUID_MAX_PRESSURE_NON_SOLVE_MS"),
});
if (jsonOnly) console.log(JSON.stringify(quiescent
  ? { lane: "quiescent", moving: summarizePowerDamPerformance(movingResult), quiescent: summary }
  : summary));
else {
  const authority = quiescent ? "quiescent paired-prefix window"
    : traceProfile ? "generic trace sample" : "throughput authority";
  console.log(`${summary.scenario}: ${summary.advanceWall_ms.toFixed(2)} ms/advance (${summary.steps} advances, ${authority})`);
  if (summary.measurementWindow) {
    console.log(`measurement window: settled through step ${summary.measurementWindow.startStep}; cumulative counters differenced over steps ${summary.measurementWindow.startStep + 1}–${summary.measurementWindow.endStep}`);
  }
  if (summary.commands) {
    console.log(`commands/advance: ${summary.commands.dispatchesPerAdvance.toFixed(1)} dispatches, ${summary.commands.computePassesPerAdvance.toFixed(1)} compute passes, ${(summary.commands.clearBytesPerAdvance / 1e6).toFixed(2)} MB clears, ${(summary.commands.copyBytesPerAdvance / 1e6).toFixed(2)} MB copies`);
    for (const [stage, passes] of Object.entries(summary.commands.computePassesByStage)
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))) {
      console.log(`compute passes/stage: ${stage}: ${passes.toFixed(1)}/advance`);
    }
    console.log(`compute-pass attribution: ${summary.commands.computePassAttributionComplete ? "complete" : `INCOMPLETE (${summary.commands.unattributedComputePassesPerAdvance.toFixed(1)} unattributed/advance)`}`);
    console.log("raw compute-pass labels are retained in --json output for contributor drill-down");
    console.log(`MGPCG: ${summary.commands.mgpcgDispatchesPerAdvance.toFixed(1)} dispatches/advance (${(100 * summary.commands.mgpcgDispatchFraction).toFixed(1)}% of all dispatches)`);
  }
  if (summary.physicsTrace) {
    const trace = summary.physicsTrace;
    console.log(`physics trace: ${trace.total_ms.toFixed(2)} ms total; ${trace.exact ? "exact" : "INEXACT"} accounting (${trace.accounted_ms.toFixed(2)} ms attributed) · ${trace.measurementSource ?? "unknown source"}`);
    for (const phase of trace.phases) {
      console.log(`physics phase: ${phase.id} · ${phase.label}: ${phase.duration_ms.toFixed(2)} ms`);
    }
  }
  if (summary.terminalCounters) {
    const counters = summary.terminalCounters;
    if (counters.activeSamples !== undefined || counters.activeFineBricks !== undefined) {
      console.log(`terminal active set: ${counters.activeSamples ?? "unavailable"} samples, ${counters.activeFineBricks ?? "unavailable"}/${counters.desiredFineBricks ?? "unavailable"} active/desired fine bricks, ${counters.transportSegments ?? "unavailable"} transport segments`);
    }
    if (counters.fineBandOccupancy !== undefined) {
      console.log(`fine band occupancy: ${(counters.fineBandOccupancy * 100).toFixed(1)}% (${counters.activeFineBricks ?? "?"} active / ${counters.logicalFineBricks ?? "?"} logical bricks, capacity ${counters.fineBrickCapacity ?? "?"})`);
    }
    if (counters.pressureIterationsExecuted !== undefined
      || counters.pressureIterationsScheduled !== undefined) {
      console.log(`terminal pressure iterations: ${counters.pressureIterationsExecuted ?? "unavailable"}/${counters.pressureIterationsScheduled ?? "unavailable"} executed/scheduled (hard limit ${counters.pressureIterationsHardLimit ?? "unavailable"})`);
    }
  }
  if (summary.fineTimestamps) {
    console.log(`fine GPU timestamps: ${summary.fineTimestamps.measuredPasses} passes measured, ${summary.fineTimestamps.invalidPasses} invalid; ${summary.fineTimestamps.summedPassPerAdvance_ms.toFixed(2)} ms/advance summed pass occupancy`);
    for (const [label, bucket] of Object.entries(summary.fineTimestamps.byLabel)
      .sort((left, right) => right[1].totalPerAdvance_ms - left[1].totalPerAdvance_ms)
      .slice(0, 20)) {
      console.log(`GPU pass: ${label}: ${bucket.totalPerAdvance_ms.toFixed(3)} ms/advance · ${bucket.mean_ms.toFixed(3)} ms mean · ${bucket.samples} samples`);
    }
  }
  if (summary.dataFlow) {
    const dispatches = summary.dataFlow.passes.reduce((sum, pass) => sum + pass.dispatches, 0);
    console.log(`GPU data flow: schema v${summary.dataFlow.schemaVersion} · ${summary.dataFlow.passes.length} pass labels · ${summary.dataFlow.buffers.length} buffers · ${dispatches} sampled dispatches`);
    console.log("GPU data flow detail is retained in --json output; byte counts are binding-range upper bounds");
  }
  console.log(`validation errors: ${summary.validationErrorCount}`);
  for (const failure of failures) console.error(`performance gate: ${failure}`);
  console.log(`acceptance: benchmark:power-dam-ui ${lane === "ui" ? "recorded above" : "required separately"}; zero validation errors ${summary.validationErrorCount === 0 ? "PASS" : "FAIL"}; performance gates ${failures.length === 0 ? "PASS" : "FAIL"}`);
  console.log("acceptance: run npm run acceptance:power-liquids-phase for the UI profile, exact two-step smoke, and 500-step minimal gate");
}
if (summary.validationErrorCount > 0 || failures.length > 0) process.exitCode = 1;
