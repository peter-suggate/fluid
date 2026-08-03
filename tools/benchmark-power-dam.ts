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
  powerDamLaneWithDt,
  powerDamLaneWithSteps,
  type PowerDamRuntimeLane as RuntimeLane,
} from "./power-dam-lane-environment";
import { POWER_DAM_PRESSURE_KERNEL_LABEL_PREFIXES } from "./power-dam-pressure-kernel-profile";
import { formatPassBrokerBoundaryAudit } from "../lib/webgpu-pass-broker";
import {
  formatResidentMemoryReport,
  type GPUResidentMemoryReport,
} from "./webgpu-smoke-gpu-audits";
import { powerDamRunEnvironment } from "./power-dam-run-environment";

const args = new Set(process.argv.slice(2));
const requestedLane = process.argv.find((argument) => argument.startsWith("--lane="))
  ?.slice("--lane=".length) ?? "mini";
const quiescent = args.has("--quiescent") || requestedLane === "quiescent";
if (requestedLane !== "quiescent" && !(requestedLane in POWER_DAM_LANE_ENVIRONMENT)) {
  throw new Error(`--lane must be quiescent or one of ${Object.keys(POWER_DAM_LANE_ENVIRONMENT).join(", ")}`);
}
if (quiescent && requestedLane === "ui") {
  throw new Error("--quiescent uses the mini dam and cannot be combined with --lane=ui");
}
const lane = (requestedLane === "quiescent" ? "moving-interface" : requestedLane) as RuntimeLane;
const traceProfile = args.has("--profile");
const fineTimestamps = args.has("--fine-timestamps");
const pressureKernelProfile = args.has("--pressure-kernel-profile");
const passTimestamps = args.has("--pass-timestamps") || args.has("--algorithm-diagnostics")
  || pressureKernelProfile;
// Superseded by --isolate-pass-labels, and it never worked: splitting Metal
// encoders cannot separate dispatches WebGPU recorded into ONE pass, which is
// what the broker does. Kept because its +0.9 ms on a 168-pass frame is a
// direct measurement of per-encoder switch cost.
const isolatePassEncoders = args.has("--isolate-passes");
if (isolatePassEncoders && !passTimestamps) {
  throw new Error("--isolate-passes requires --pass-timestamps; on its own it only makes the frame slower");
}
// The attribution mode that works. Every labelled `compute()` gets its own
// pass, so a label's timestamps bracket that label's dispatches and nothing
// else. Costs extra pass launches, so the frame is inflated: read the ranking,
// never the ms/advance.
const isolatePassLabels = args.has("--isolate-pass-labels") || pressureKernelProfile;
if (isolatePassLabels && !passTimestamps) {
  throw new Error("--isolate-pass-labels requires --pass-timestamps; on its own it only makes the frame slower");
}
const pressureMicroStages = args.has("--pressure-micro-stages");
if (pressureMicroStages && (!passTimestamps || !isolatePassLabels)) {
  throw new Error("--pressure-micro-stages requires --pass-timestamps and --isolate-pass-labels");
}
if (pressureMicroStages && pressureKernelProfile) {
  throw new Error("--pressure-micro-stages and --pressure-kernel-profile select different timestamp scopes");
}
const pressureTimestampPrefixes = pressureMicroStages
  ? ["SPGrid accurate A2 -", "SPGrid Section 6.3 -"]
  : pressureKernelProfile ? [...POWER_DAM_PRESSURE_KERNEL_LABEL_PREFIXES] : [];
if (quiescent && fineTimestamps) {
  throw new Error("--quiescent cannot use --fine-timestamps until the runner can timestamp only a trailing window");
}
const jsonOnly = args.has("--json");
const forwardNDJSON = args.has("--forward-ndjson");
// Discovery-only escape hatch. Tripwires still execute and every failure is
// retained loudly, but an intentionally quality-invalid ablation may finish
// far enough to report its wall-clock lower bound.
const qualityInvalidProbe = args.has("--quality-invalid-probe");
const artifactPath = process.argv.find((argument) => argument.startsWith("--artifact="))
  ?.slice("--artifact=".length);
const diagnosticArtifactPath = process.argv.find((argument) =>
  argument.startsWith("--diagnostic-artifact="))?.slice("--diagnostic-artifact=".length);
const root = fileURLToPath(new URL("..", import.meta.url));
const runner = fileURLToPath(new URL("./run-webgpu-smoke-isolated.ts", import.meta.url));

const laneEnvironment = POWER_DAM_LANE_ENVIRONMENT;

/**
 * Everything this harness and the lane table assert on top of the caller's
 * shell. Kept separate from the merge below so the artifact can name which
 * variables the run *inherited* rather than chose — the C7 ambiguity in
 * `docs/STRUCTURED_CUTOVER_LEDGER.md` was unresolvable precisely because a
 * merged environment cannot distinguish the two.
 */
const benchmarkEnvironmentOverlay = (overrides: Record<string, string> = {}): Record<string, string> => ({
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
    FLUID_GPU_ISOLATE_PASS_ENCODERS: isolatePassEncoders ? "1" : "0",
    FLUID_GPU_ISOLATE_PASS_LABELS: isolatePassLabels ? "1" : "0",
    // One captured command buffer is one advance, and one advance is not a
    // stable sample: back-to-back single-buffer captures reordered the top of
    // the table and moved individual stages by more than 4x. Twenty-five
    // captures, taken after the cold opening advances, reproduced the rank
    // order exactly and every stage to better than 1% across two runs. The
    // caller can still pin either number for a targeted capture.
    ...(isolatePassLabels ? {
      FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS:
        process.env.FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS ?? "25",
      FLUID_GPU_PASS_TIMESTAMP_SKIP_COMMAND_BUFFERS:
        process.env.FLUID_GPU_PASS_TIMESTAMP_SKIP_COMMAND_BUFFERS ?? "40",
      ...(pressureTimestampPrefixes.length > 0 ? {
        // Retain every split A2 stage while declining unrelated labels before
        // they consume query slots. Scope PassBroker isolation to those same
        // prefixes: unrelated work keeps the production pass graph, reducing
        // both launch perturbation and timestamp capacity pressure.
        FLUID_GPU_PASS_TIMESTAMP_LABEL_PREFIXES:
          pressureTimestampPrefixes.join(","),
        FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES:
          pressureTimestampPrefixes.join(","),
      } : {}),
    } : {}),
    FLUID_ALGORITHM_DIAGNOSTICS: args.has("--algorithm-diagnostics") ? "1" : "0",
    FLUID_GPU_COMMAND_AUDIT: "1",
    // Throughput profiles keep the lightweight per-step tripwires below, but
    // must not inherit the scene catalog's full generation audit. That audit
    // also owns the four-pass structured energy probe; inheriting only its
    // host-side validation would reject the deliberately unprobed samples.
    FLUID_POWER_GENERATION_AUDIT: "0",
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
    //
    // The caller may only *escalate*, to `failfast`, which terminates at the
    // step that tripped instead of after the measured window. That is the mode
    // to diagnose a red lane in; it is not the mode to measure in, because its
    // per-step queue fence removes host/GPU overlap and cost +26.8% on the
    // large lane. Any `FLUID_TRIPWIRES=0` from the environment is ignored here
    // by construction -- this assignment follows `...overrides`.
    FLUID_TRIPWIRES: process.env.FLUID_TRIPWIRES === "failfast" ? "failfast" : "1",
    ...(qualityInvalidProbe ? { FLUID_TRIPWIRE_ALLOW:
      "topology-rollback,restriction-unaccepted,mgpcg-nonconvergence,fine-band-sentinel",
      FLUID_TRIPWIRE_ALLOW_SUMMARY: "1", FLUID_QUALITY_INVALID_PROBE: "1" } : {}),
});

const benchmarkEnvironment = (overrides: Record<string, string> = {}): NodeJS.ProcessEnv => ({
  ...process.env, ...benchmarkEnvironmentOverlay(overrides),
});

/** The overlay of the last run started, for the artifact's environment record. */
let lastAuthoredOverlay: Record<string, string> = benchmarkEnvironmentOverlay();
let lastResolvedEnvironment: NodeJS.ProcessEnv = benchmarkEnvironment();

const runBenchmark = async (overrides: Record<string, string> = {}): Promise<PowerDamResultRecord> => {
  lastAuthoredOverlay = benchmarkEnvironmentOverlay(overrides);
  lastResolvedEnvironment = { ...process.env, ...lastAuthoredOverlay };
  const child = spawn(process.execPath, ["--import", "tsx", runner], {
    cwd: root,
    env: lastResolvedEnvironment,
    stdio: ["ignore", "pipe", "inherit"],
  });
  let result: PowerDamResultRecord | undefined;
  /** Correctness verdicts are forwarded unconditionally; census/progress records
   * stay behind `--forward-ndjson`.
   *
   * The child's stdout is consumed by this reader, so anything not re-printed is
   * discarded. That silently ate the touched-directory differential: a run with
   * `FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE=1` emitted its verdict, this harness
   * swallowed it, and the run reported "performance gates PASS" with the
   * correctness question never answered on screen. A verdict that only exists
   * inside a pipe is not evidence. */
  const CORRECTNESS_PHASES = new Set([
    "spgrid-touched-directory-tripwire", "tripwires",
  ]);
  const lines = createInterface({ input: child.stdout! });
  lines.on("line", (line) => {
    result = powerDamResultFromLine(line) ?? result;
    try {
      const record = JSON.parse(line) as {
        record?: string; phase?: string; when?: string;
        residentMemory?: GPUResidentMemoryReport;
      };
      if (record.phase !== undefined && CORRECTNESS_PHASES.has(record.phase)) {
        console.log(line);
        return;
      }
      // Unconditional for the same reason as a correctness verdict: the census
      // only runs when `FLUID_GPU_MEMORY_CENSUS` explicitly asked for it, and a
      // census you requested and cannot see is the failure this reader's own
      // comment describes. Forwarded with the summary the child cannot print.
      if (record.phase === "gpu-memory" && record.residentMemory) {
        console.log(line);
        console.log(`--- GPU resident memory (${record.when}) ---`);
        console.log(formatResidentMemoryReport(record.residentMemory));
        return;
      }
      if (forwardNDJSON
        && (record.record === "progress"
          || record.phase === "octree-row-delta-census-sample"
          || record.phase === "frame-redundancy-census"
          || record.phase === "fine-transport-workset-census"
          || record.phase === "structured-workset-census"
          || record.phase === "settled-maintenance-census")) {
        console.log(line);
      }
    } catch { /* forward only machine-readable experiment records */ }
  });
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

// Ablation lane. Wall clock still SIZES a saving: `--pass-timestamps
// --isolate-pass-labels` ranks stages honestly but reads ~19% high in
// aggregate (48.6 ms of brackets against a 39.2 ms clean advance), because the
// one captured command buffer carries every timestamp write.
// So A/B iteration needs it cheap. `--steps=N` shortens the lane without
// desynchronising FLUID_TARGET_S from FLUID_MAX_DT. Compare only runs at the
// SAME step count: the frame drifts materially over a lane, so a 150-step mean
// and a 500-step mean describe different regimes.
const stepsOverride = process.argv.find((argument) => argument.startsWith("--steps="))
  ?.slice("--steps=".length);
const dtOverride = process.argv.find((argument) => argument.startsWith("--dt="))
  ?.slice("--dt=".length);
const targetOverride = process.argv.find((argument) => argument.startsWith("--target-s="))
  ?.slice("--target-s=".length);
const bandLevelOverride = process.argv.find((argument) => argument.startsWith("--band-level="))
  ?.slice("--band-level=".length);
const fineFactorOverride = process.argv.find((argument) => argument.startsWith("--fine-factor="))
  ?.slice("--fine-factor=".length);
const leafSizeOverride = process.argv.find((argument) => argument.startsWith("--leaf-size="))
  ?.slice("--leaf-size=".length);
if (stepsOverride !== undefined && dtOverride !== undefined) {
  throw new Error("--steps and --dt are mutually exclusive; --dt derives steps at fixed simulated time");
}
const laneOverride: Record<string, string> = dtOverride !== undefined ? (() => {
  if (quiescent) throw new Error("--dt cannot override the quiescent paired-prefix lane");
  const dt = Number(dtOverride);
  const target = targetOverride === undefined ? 2 : Number(targetOverride);
  return powerDamLaneWithDt(lane, dt, target);
})() : stepsOverride === undefined ? {} : (() => {
  const steps = Number(stepsOverride);
  if (!Number.isInteger(steps) || steps <= 0) throw new Error(`--steps must be a positive integer; received ${stepsOverride}`);
  if (quiescent) throw new Error("--steps cannot override the quiescent lane, which owns its own settle/measure split");
  return powerDamLaneWithSteps(lane, steps);
})();
const bandEnvironmentOverride: Record<string, string> = {};
if (bandLevelOverride !== undefined) {
  const bandLevel = Number(bandLevelOverride);
  if (!Number.isInteger(bandLevel) || bandLevel < 0 || bandLevel > 4) {
    throw new Error(`--band-level must be an integer from 0 through 4; received ${bandLevelOverride}`);
  }
  // Applied after the authored lane so an A/B cannot silently fall back to
  // the mini lane's recorded level-3 default.
  bandEnvironmentOverride.FLUID_OCTREE_INTERFACE_BAND = String(bandLevel);
}
if (fineFactorOverride !== undefined) {
  const fineFactor = Number(fineFactorOverride);
  if (![1, 4, 8].includes(fineFactor)) {
    throw new Error(`--fine-factor must be one of 1, 4, or 8; received ${fineFactorOverride}`);
  }
  // Applied after the authored lane for differential Bet-4 measurements.
  // This changes the discretization and therefore always requires Gate B.
  bandEnvironmentOverride.FLUID_OCTREE_GLOBAL_FINE_FACTOR = String(fineFactor);
}
if (leafSizeOverride !== undefined) {
  const leafSize = Number(leafSizeOverride);
  if (!Number.isInteger(leafSize) || leafSize < 1 || (leafSize & (leafSize - 1)) !== 0) {
    throw new Error(`--leaf-size must be a positive power of two; received ${leafSizeOverride}`);
  }
  // The lane table pins `FLUID_MAXIMUM_LEAF_SIZE`, so a shell export cannot
  // reach it and an `--arm=` cannot either. This override exists because the
  // lane's own leaf size has already changed under the program once
  // (`065219a`, mini 2 -&gt; 32) and the artifacts on either side were compared as
  // if they described the same lane. Changing it changes the discretization:
  // Gate B, and never a number to quote next to a default-lane capture.
  bandEnvironmentOverride.FLUID_MAXIMUM_LEAF_SIZE = String(leafSize);
}

const movingResult = await runBenchmark({ ...laneOverride, ...bandEnvironmentOverride });
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
  const settlePrefix = await runBenchmark({
    ...environmentForSteps(settleSteps), ...bandEnvironmentOverride,
  });
  const complete = await runBenchmark({
    ...environmentForSteps(settleSteps + measuredSteps), ...bandEnvironmentOverride,
  });
  terminalResult = complete;
  artifactResult = powerDamResultWindow(settlePrefix, complete);
  const quiescentSummary = summarizePowerDamPerformance(artifactResult);
  if (!jsonOnly) {
    console.log(`${summary.scenario}: ${summary.advanceWall_ms.toFixed(2)} ms/advance (${summary.steps} advances, in motion)`);
  }
  summary = quiescentSummary;
}
// Resolved once, from the last run's overlay, and written into every artifact
// this invocation produces. Ledger C7: an artifact without its environment
// cannot be compared to anything, including its own re-run.
const runEnvironment = powerDamRunEnvironment({
  lane: requestedLane,
  argv: process.argv.slice(2),
  resolved: lastResolvedEnvironment as Record<string, string | undefined>,
  authored: lastAuthoredOverlay,
  repositoryRoot: root,
});
if (diagnosticArtifactPath) {
  const absoluteDiagnosticPath = resolve(root, diagnosticArtifactPath);
  mkdirSync(dirname(absoluteDiagnosticPath), { recursive: true });
  writeFileSync(absoluteDiagnosticPath, `${JSON.stringify({
    schemaVersion: 2,
    lane: requestedLane,
    capturedAt: new Date().toISOString(),
    environment: runEnvironment,
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
    environment: runEnvironment,
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
  const output = quiescent
    ? { lane: "quiescent", environment: runEnvironment,
      moving: summarizePowerDamPerformance(movingResult), quiescent: summary }
    : { ...summary, environment: runEnvironment };
  console.log(JSON.stringify(qualityInvalidProbe
    ? { ...output, qualityInvalidProbe: true, probeFailures: failures }
    : output));
} else {
  const authority = quiescent ? "quiescent paired-prefix window"
    : traceProfile ? "generic trace sample" : "throughput authority";
  console.log(`${summary.scenario}: ${summary.advanceWall_ms.toFixed(2)} ms/advance (${summary.steps} advances, ${authority})`);
  // Printed next to the number it qualifies, not buried in the artifact. A
  // contaminated wall compared against a clean one is ledger C7 happening again.
  console.log(`environment: ${runEnvironment.clean ? "measurement-clean" : "CONTAMINATED"}`
    + ` · comparison key ${runEnvironment.comparisonKey}`
    + ` · ${runEnvironment.git.head?.slice(0, 7) ?? "no-git"}${runEnvironment.git.dirty ? "-dirty" : ""}`
    + ` · ${runEnvironment.adapter}`);
  for (const entry of runEnvironment.contaminants) {
    console.log(`environment contaminant: ${entry.name}=${entry.resolved ?? "(unset)"}`
      + ` (measurement-clean: ${entry.clean.length === 0 ? "unclassified shell variable"
        : entry.clean.map((value) => value ?? "(unset)").join(" | ")})`);
  }
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
    // Every ms below is divided by the captured command buffers, so the table
    // reads per advance no matter how many advances were sampled.
    const captures = Math.max(1, timestamps.capturedCommandBuffers);
    console.log(`compute-pass GPU timestamps: ${timestamps.measuredPasses} passes in ${timestamps.capturedCommandBuffers} command buffer(s), ${timestamps.invalidPasses} invalid, ${timestamps.capacityOverflows} capacity overflows; ${(timestamps.summedPass_ms / captures).toFixed(2)} ms/advance summed pass occupancy over a ${((timestamps.span_ms ?? 0) / captures).toFixed(2)} ms/advance GPU span (coverage ${(timestamps.coverageRatio ?? 0).toFixed(3)})`);
    // The coverage ratio is the licence to read the table at all. Passes tile
    // the command buffer only when each labelled `compute()` owns its pass;
    // otherwise a label is its group's total charged to whichever pass stayed
    // open, which is how a 0.3 ms workset publication once read as 3.6 ms.
    console.log(timestamps.labelIsolated
      ? timestamps.labelPrefixes && timestamps.labelPrefixes.length > 0
        ? `compute-pass attribution: per-LABEL FILTERED — every retained compute() owned its pass; ${timestamps.measuredPasses} samples match [${timestamps.labelPrefixes.join(", ")}], with ${timestamps.capacityOverflows} dropped for capacity. Per-label times and call counts are exact; coverage is intentionally partial because unrelated stages were not timestamped`
        : `compute-pass attribution: per-LABEL — every labelled compute() owned its pass, and the brackets tile ${((timestamps.coverageRatio ?? 0) * 100).toFixed(1)}% of the GPU span. Ranking is trustworthy; absolute ms is inflated by the extra pass launches, so size a saving by ablation on the wall clock (--steps=N)`
      : timestamps.encoderIsolated
        ? "compute-pass attribution: encoder-isolated, and STILL NOT per-pass — splitting Metal encoders does not stop a pass staying open across later stages; add --isolate-pass-labels"
        : "compute-pass attribution: per-GROUP, not per-pass — PassBroker keeps one pass open across many compute({label}) calls and drops the later labels, so each label below is everything encoded until the next fence; add --isolate-pass-labels, or attribute by ablation on the wall clock (--steps=N)");
    for (const [label, bucket] of Object.entries(timestamps.byLabel)
      .sort((left, right) => right[1].total_ms - left[1].total_ms)
      .slice(0, 40)) {
      console.log(`GPU compute pass: ${label}: ${(bucket.total_ms / captures).toFixed(3)} ms/advance · ${bucket.mean_ms.toFixed(3)} ms mean · ${bucket.samples} samples`);
    }
  }
  if (summary.pressureKernelProfile) {
    const profile = summary.pressureKernelProfile;
    console.log(`pressure-kernel GPU attribution: ${profile.exact ? "exact" : "INCOMPLETE"}; ${profile.instrumentedPressurePerAdvance_ms.toFixed(3)} ms/instrumented advance across ${profile.capturedAdvances} captures`);
    for (const [region, bucket] of Object.entries(profile.byRegion)) {
      console.log(`pressure region: ${region}: ${bucket.totalPerAdvance_ms.toFixed(3)} ms/advance · ${(100 * bucket.shareOfPressure).toFixed(1)}% · ${bucket.samples} samples`);
    }
    for (const warning of profile.warnings) console.log(`pressure profile note: ${warning}`);
  }
  if (summary.passBoundaries) {
    const boundaries = summary.passBoundaries;
    // The pass budget is a boundary problem: a compute pass exists because
    // something fenced. `computePassesPerAdvance` above says how many passes
    // there are; this says WHY each one ended, so a merge can be aimed.
    console.log(`pass boundaries/advance: ${boundaries.passClosuresPerAdvance.toFixed(1)} closures`
      + ` from ${boundaries.requestsPerAdvance.toFixed(1)} fence requests`
      + ` (${boundaries.idempotentRequestsPerAdvance.toFixed(1)} idempotent — already batched)`
      + ` · ${boundaries.copyCommandsPerAdvance.toFixed(1)} copies`
      + ` · ${boundaries.clearCommandsPerAdvance.toFixed(1)} clears`
      + ` · ${(boundaries.bytesPerAdvance / 1e6).toFixed(2)} MB`
      + ` · ${boundaries.brokersPerAdvance.toFixed(1)} command encoders`);
    // Both honesty flags print next to the table, never only in the artifact: a
    // label-isolated stream carries a synthetic boundary per label change, and a
    // census that still holds construction inflates every number above.
    // The ranked table below is RAW totals over the measured window; only the
    // line above is per advance. Say so, next to both.
    console.log(`pass boundary census: ${boundaries.exact ? "exact" : "QUALIFIED"}`
      + ` · totals below cover ${boundaries.measuredAdvances} measured advances`
      + ` · warmup ${boundaries.warmupExcluded ? "excluded" : "INCLUDED"}`
      + ` · FLUID_GPU_ISOLATE_PASS_LABELS ${boundaries.labelIsolated
        ? `ON for ${boundaries.labelIsolatedBrokers}/${boundaries.brokers} encoders`
        : "off"}`
      + ` · ${boundaries.absorbedLabelPairs} stage labels already share a pass with another`
      + ` stage across ${boundaries.compositeEncoderLabels} composite encoders`);
    for (const line of formatPassBrokerBoundaryAudit(
      new Map(Object.entries(boundaries.byReason)), 20)) {
      console.log(`pass boundary: ${line}`);
    }
    for (const warning of boundaries.warnings) console.log(`pass boundary note: ${warning}`);
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
if (summary.validationErrorCount > 0 || (!qualityInvalidProbe && failures.length > 0)) process.exitCode = 1;
