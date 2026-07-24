import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import {
  powerDamResultFromLine,
  powerDamPerformanceFailures,
  summarizePowerDamPerformance,
  type PowerDamResultRecord,
} from "./power-dam-performance-report";

type Lane = "mini" | "ui";

const args = new Set(process.argv.slice(2));
const laneValue = process.argv.find((argument) => argument.startsWith("--lane="))?.slice("--lane=".length) ?? "mini";
if (laneValue !== "mini" && laneValue !== "ui") throw new Error("--lane must be mini or ui");
const lane = laneValue as Lane;
const traceProfile = args.has("--profile");
const jsonOnly = args.has("--json");
const root = fileURLToPath(new URL("..", import.meta.url));
const runner = fileURLToPath(new URL("./run-webgpu-smoke-isolated.ts", import.meta.url));

const laneEnvironment: Record<Lane, Record<string, string>> = {
  mini: {
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

const child = spawn(process.execPath, ["--import", "tsx", runner], {
  cwd: root,
  env: {
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
    FLUID_GPU_COMMAND_AUDIT: traceProfile ? "0" : "1",
    FLUID_CPU_ORACLE: "0", FLUID_FIELD_STATS: "0", FLUID_SPARSE_STATS: "0",
    FLUID_RASTER_CHECKPOINTS: "0", FLUID_WEBGPU_SMOKE_TIMEOUT_MS: "240000",
    ...laneEnvironment[lane],
  },
  stdio: ["ignore", "pipe", "inherit"],
});

let result: PowerDamResultRecord | undefined;
const lines = createInterface({ input: child.stdout! });
lines.on("line", (line) => { result = powerDamResultFromLine(line) ?? result; });

const exitCode = await new Promise<number>((resolve, reject) => {
  child.once("error", reject);
  child.once("exit", (code, signal) => signal ? reject(new Error(`Dawn benchmark exited from ${signal}`)) : resolve(code ?? 1));
});
if (exitCode !== 0) process.exit(exitCode);
if (!result) throw new Error("Dawn benchmark completed without a result record");

const summary = summarizePowerDamPerformance(result);
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
if (jsonOnly) console.log(JSON.stringify(summary));
else {
  const authority = traceProfile ? "generic trace sample" : "throughput authority";
  console.log(`${summary.scenario}: ${summary.advanceWall_ms.toFixed(2)} ms/advance (${summary.steps} advances, ${authority})`);
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
    console.log(`physics trace: ${trace.total_ms.toFixed(2)} ms total; ${trace.exact ? "exact" : "INEXACT"} accounting (${trace.accounted_ms.toFixed(2)} ms attributed)`);
    for (const phase of trace.phases) {
      console.log(`physics phase: ${phase.id} · ${phase.label}: ${phase.duration_ms.toFixed(2)} ms`);
    }
  }
  console.log(`validation errors: ${summary.validationErrorCount}`);
  for (const failure of failures) console.error(`performance gate: ${failure}`);
  console.log(`acceptance: benchmark:power-dam-ui ${lane === "ui" ? "recorded above" : "required separately"}; zero validation errors ${summary.validationErrorCount === 0 ? "PASS" : "FAIL"}; performance gates ${failures.length === 0 ? "PASS" : "FAIL"}`);
  console.log("acceptance: run npm run acceptance:power-liquids-phase for the UI profile, exact two-step smoke, and 500-step minimal gate");
}
if (summary.validationErrorCount > 0 || failures.length > 0) process.exitCode = 1;
