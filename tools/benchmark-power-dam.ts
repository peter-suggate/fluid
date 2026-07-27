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
import {
  POWER_DAM_LANE_ENVIRONMENT,
  type PowerDamRuntimeLane as RuntimeLane,
} from "./power-dam-lane-environment";

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
const passTimestamps = args.has("--pass-timestamps") || args.has("--algorithm-diagnostics");
if (quiescent && fineTimestamps) {
  throw new Error("--quiescent cannot use --fine-timestamps until the runner can timestamp only a trailing window");
}
const jsonOnly = args.has("--json");
const artifactPath = process.argv.find((argument) => argument.startsWith("--artifact="))
  ?.slice("--artifact=".length);
const diagnosticArtifactPath = process.argv.find((argument) =>
  argument.startsWith("--diagnostic-artifact="))?.slice("--diagnostic-artifact=".length);
const root = fileURLToPath(new URL("..", import.meta.url));
const runner = fileURLToPath(new URL("./run-webgpu-smoke-isolated.ts", import.meta.url));

const laneEnvironment = POWER_DAM_LANE_ENVIRONMENT;

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
    FLUID_GPU_PASS_TIMESTAMPS: passTimestamps ? "1" : "0",
    FLUID_ALGORITHM_DIAGNOSTICS: args.has("--algorithm-diagnostics") ? "1" : "0",
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
    // Silent-failure tripwires are unconditional on every benchmark lane and
    // cannot be switched off by a lane table or a caller override: a run that
    // gets faster while rolling back topology, leaving restriction rows
    // unaccepted, overflowing the fine band, or publishing the seed pressure
    // is not a speedup. `FLUID_TRIPWIRES=1` also makes an *unevaluable*
    // counter a failure. See docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3.
    FLUID_TRIPWIRES: "1",
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
// The raw record of the run whose FINAL advance the terminal gates describe.
// `powerDamResultWindow` rebuilds a differenced record from an explicit field
// list, so per-run tripwire counters must come from the unwindowed record.
let terminalResult: PowerDamResultRecord = movingResult;
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
  terminalResult = complete;
  artifactResult = powerDamResultWindow(settlePrefix, complete);
  const quiescentSummary = summarizePowerDamPerformance(artifactResult);
  if (!jsonOnly) {
    console.log(`${summary.scenario}: ${summary.advanceWall_ms.toFixed(2)} ms/advance (${summary.steps} advances, in motion)`);
  }
  summary = quiescentSummary;
}
if (diagnosticArtifactPath) {
  const absoluteDiagnosticPath = resolve(root, diagnosticArtifactPath);
  mkdirSync(dirname(absoluteDiagnosticPath), { recursive: true });
  writeFileSync(absoluteDiagnosticPath, `${JSON.stringify({
    schemaVersion: 1,
    lane: requestedLane,
    capturedAt: new Date().toISOString(),
    summary,
  }, null, 2)}\n`);
  if (!jsonOnly) console.log(`diagnostic artifact: ${absoluteDiagnosticPath}`);
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
const failures = [...powerDamPerformanceFailures(summary, {
  maximumAdvanceWall_ms: numericLimit("FLUID_MAX_ADVANCE_MS"),
  maximumDispatchesPerAdvance: numericLimit("FLUID_MAX_DISPATCHES_PER_ADVANCE"),
  maximumComputePassesPerAdvance: numericLimit("FLUID_MAX_PASSES_PER_ADVANCE"),
  maximumPressureNonSolve_ms: numericLimit("FLUID_MAX_PRESSURE_NON_SOLVE_MS"),
})];
// ---- Per-run silent-failure tripwires -------------------------------------
// docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3 (P0.3). A fine band that overflows
// its capacity degrades the resident-brick count to the INVALID sentinel and
// leaves the pressure solve executing zero iterations -- yet every gate above
// still reports success, because a solver that computes nothing also produces
// no validation errors. These gates are unconditional, and a counter that is
// *unavailable* fails just as loudly as a counter that is out of range: an
// unevaluable tripwire is the failure mode this item exists to kill.
//
// (The per-generation topology-rollback / unaccepted-restriction-row and the
// per-step MGPCG convergence tripwires are enforced inside the smoke harness,
// which is the only place with per-step access to those control words.)
{
  const terminal = summary.terminalCounters;
  const active = terminal?.activeFineBricks;
  const logical = terminal?.logicalFineBricks;
  const FINE_BAND_INVALID_SENTINEL = 0xffff_ffff;
  // Tripwire 4: the worklist header's active count degraded to INVALID. The
  // count is header word ONE; a prior consumer read word zero (the generation)
  // and printed nonsense, so this reads the decoded counter, not a raw word.
  if (active === undefined) {
    failures.push("tripwire could not be evaluated: terminal fine-band active-brick count is"
      + " unavailable (globalFineActiveBricks missing from the smoke result record)");
  } else if (active === FINE_BAND_INVALID_SENTINEL) {
    failures.push(`fine band active-brick count is the INVALID sentinel ${FINE_BAND_INVALID_SENTINEL}`
      + ` (0xFFFFFFFF) -- the band overflowed its ${terminal?.fineBrickCapacity ?? "unknown"}-brick`
      + " capacity, which silently no-ops the solver while the run still reports PASS");
  }
  if (logical === undefined) {
    failures.push("tripwire could not be evaluated: logical fine-brick count is unavailable"
      + " (globalFineLevelSetLogicalBrickCount missing from the smoke result record)");
  } else if (active !== undefined && active !== FINE_BAND_INVALID_SENTINEL && active > logical) {
    failures.push(`fine band active-brick count ${active} exceeds the ${logical}-brick logical`
      + " lattice -- the band overflowed capacity and the publication is invalid");
  }
  // Tripwire 3: terminal pressure iterations. Read from two independent
  // sources -- the solver telemetry counter and the GPU MGPCG control word the
  // harness decodes after the final advance -- because the telemetry counter
  // is only as fresh as the last solve-diagnostics readback.
  const terminalMgpcg = (terminalResult as {
    octreeMGPCGDiagnostics?: {
      flags?: number; converged?: boolean; iterations?: number; rows?: number;
      relativeResidual?: number;
    };
  }).octreeMGPCGDiagnostics;
  const executed = terminal?.pressureIterationsExecuted;
  if (executed === undefined) {
    failures.push(`tripwire could not be evaluated on lane ${requestedLane}: terminal pressure`
      + " iteration count is unavailable (quadtreePressureIterationsUsed missing from the"
      + " smoke result record)");
  } else if (executed === 0) {
    failures.push(`pressure solve executed zero iterations on lane ${requestedLane}`
      + ` (scheduled ${terminal?.pressureIterationsScheduled ?? "unknown"}, hard limit`
      + ` ${terminal?.pressureIterationsHardLimit ?? "unknown"}) -- the solver did no work.`
      + " On a churn lane (mini/moving-interface/ui) this is the tripwire firing. On the"
      + " settled quiescent window a seed pressure that is already inside tolerance can"
      + " legitimately converge in zero iterations; confirm against the churn lanes before"
      + " treating it as a solver fault");
  }
  if (terminalMgpcg === undefined) {
    failures.push(`tripwire could not be evaluated on lane ${requestedLane}: the terminal MGPCG`
      + " control readback (octreeMGPCGDiagnostics) is missing from the smoke result record");
  } else {
    if (terminalMgpcg.iterations === undefined) {
      failures.push(`tripwire could not be evaluated on lane ${requestedLane}: the terminal MGPCG`
        + ` control carries no iteration count: ${JSON.stringify(terminalMgpcg)}`);
    } else if (terminalMgpcg.iterations === 0) {
      failures.push(`terminal MGPCG control reports zero executed iterations on lane`
        + ` ${requestedLane}: ${JSON.stringify(terminalMgpcg)}. See the note on the`
        + " telemetry counter above for the settled-window caveat");
    }
    // Non-convergence publishes the SEED pressure and fails nothing today.
    if (terminalMgpcg.converged !== true && (terminalMgpcg.iterations ?? 0) > 0) {
      failures.push(`terminal pressure solve did not converge on lane ${requestedLane}:`
        + ` ${JSON.stringify(terminalMgpcg)}`);
    }
  }
}
// Gate failures always reach stderr, including under --json: a tripwire whose
// counter values are only visible after a rerun is a tripwire that costs a GPU
// session to triage.
if (jsonOnly) {
  for (const failure of failures) console.error(`performance gate: ${failure}`);
  console.log(JSON.stringify(quiescent
    ? { lane: "quiescent", moving: summarizePowerDamPerformance(movingResult), quiescent: summary }
    : summary));
} else {
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
  if (summary.passTimestamps) {
    const timestamps = summary.passTimestamps;
    console.log(`compute-pass GPU timestamps: ${timestamps.measuredPasses} passes in ${timestamps.capturedCommandBuffers} command buffer(s), ${timestamps.invalidPasses} invalid, ${timestamps.capacityOverflows} capacity overflows; ${timestamps.summedPass_ms.toFixed(2)} ms summed pass occupancy`);
    for (const [label, bucket] of Object.entries(timestamps.byLabel)
      .sort((left, right) => right[1].total_ms - left[1].total_ms)
      .slice(0, 40)) {
      console.log(`GPU compute pass: ${label}: ${bucket.total_ms.toFixed(3)} ms · ${bucket.mean_ms.toFixed(3)} ms mean · ${bucket.samples} samples`);
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
