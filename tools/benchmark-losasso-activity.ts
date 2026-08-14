import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import type { FineLevelSetActivityCensus }
  from "../lib/core/fine-levelset-activity-census";

interface Arm {
  readonly id: string;
  readonly scene: string;
  readonly lane: string;
  readonly steps: number;
}

interface CensusLine {
  readonly scenario: string;
  readonly method: string;
  readonly phase: "fine-activity-census";
  readonly samples: readonly FineLevelSetActivityCensus[];
}

interface ArmRun {
  readonly census: CensusLine;
  readonly gate: "passed" | "failed";
}

const defaultArms: readonly Arm[] = [
  // The current baseline first rejects its fine publication at step 170 on a
  // 500-step probe. Keep the activity guide inside the last green interval;
  // the authored settled tank below is the calm lane until that independent
  // long-tail correctness cliff is repaired.
  { id: "symmetric-expansion-early-late", scene: "symmetric-expansion", lane: "default", steps: 150 },
  // A 250-step probe currently reaches the hard MGPCG ceiling at its terminal
  // receipt. Step 180 is the last reported green boundary on this baseline.
  { id: "settled-tank", scene: "settled-tank", lane: "acceptance", steps: 180 },
  { id: "minimal-dam-late", scene: "minimal-power-dam-break", lane: "performance", steps: 500 },
  { id: "ocean-seiche", scene: "ocean-seiche", lane: "global-fine-one-step", steps: 250 },
];

const selected = new Set((process.env.FLUID_LOSASSO_ACTIVITY_ARMS ?? "")
  .split(",").map((value) => value.trim()).filter(Boolean));
const configuredArms = selected.size === 0 ? defaultArms
  : defaultArms.filter((arm) => selected.has(arm.id));
const stepOverrideText = process.env.FLUID_LOSASSO_ACTIVITY_STEPS;
const stepOverride = stepOverrideText === undefined ? undefined : Number(stepOverrideText);
if (stepOverride !== undefined) {
  assert.ok(Number.isSafeInteger(stepOverride) && stepOverride > 0,
    "FLUID_LOSASSO_ACTIVITY_STEPS must be a positive integer");
}
const arms = configuredArms.map((arm) => stepOverride === undefined
  ? arm : { ...arm, steps: stepOverride });
assert.ok(arms.length > 0, `FLUID_LOSASSO_ACTIVITY_ARMS selected no known arm; expected ${
  defaultArms.map((arm) => arm.id).join(", ")}`);

const run = (arm: Arm) => new Promise<ArmRun>((resolve, reject) => {
  const child = spawn(process.execPath,
    ["--import", "tsx", "tools/run-webgpu-smoke-isolated.ts"], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        FLUID_SCENE: arm.scene,
        FLUID_LANE: arm.lane,
        FLUID_METHOD: "losasso",
        FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
        FLUID_TARGET_S: String(arm.steps * 0.004),
        FLUID_MAX_DT: "0.004",
        FLUID_ORACLE_STEPS: String(arm.steps),
        FLUID_EXPECT_EXACT_STEPS: String(arm.steps),
        FLUID_FIELD_STATS: "0",
        FLUID_STABILITY_ENVELOPE: "0",
        FLUID_CHECKPOINT_EVERY_S: "0",
        FLUID_ENERGY_EVERY_STEPS: "0",
        FLUID_REQUIRE_SPATIAL_FIELD: "0",
        FLUID_RASTER_CHECKPOINTS: "0",
        FLUID_PERFORMANCE_PROFILE: "1",
        FLUID_GPU_COMMAND_AUDIT: "0",
        FLUID_TRIPWIRES: "1",
        FLUID_FINE_ACTIVITY_CENSUS: "1",
        FLUID_WEBGPU_SMOKE_TIMEOUT_MS: process.env.FLUID_WEBGPU_SMOKE_TIMEOUT_MS ?? "240000",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
  child.stderr.pipe(process.stderr);
  let pending = "";
  const tail: string[] = [];
  let census: CensusLine | undefined;
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    pending += chunk;
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) {
      tail.push(line);
      if (tail.length > 24) tail.shift();
      try {
        const parsed = JSON.parse(line) as Partial<CensusLine>;
        if (parsed.phase === "fine-activity-census") census = parsed as CensusLine;
      } catch { /* Non-JSON child diagnostics stay available in the failure tail. */ }
    }
  });
  child.on("error", reject);
  child.on("exit", (code, signal) => {
    if (!census) {
      reject(new Error(`${arm.id} activity census failed (${signal ?? code ?? "unknown"})\n${tail.join("\n")}`));
      return;
    }
    resolve({ census, gate: code === 0 ? "passed" : "failed" });
  });
});

const rounded = (value: number, digits = 4) => Number(value.toFixed(digits));
const summarize = (samples: readonly FineLevelSetActivityCensus[]) => {
  const valid = samples.filter((sample) => sample.receiptValid);
  const mean = (select: (sample: FineLevelSetActivityCensus) => number) =>
    valid.reduce((sum, sample) => sum + select(sample), 0) / Math.max(1, valid.length);
  return {
    advances: samples.length,
    invalidReceipts: samples.length - valid.length,
    livePages: rounded(mean((sample) => sample.liveBandPages), 1),
    dirty: rounded(mean((sample) => sample.dirtyFraction)),
    dirtyHalo: rounded(mean((sample) => sample.dirtyHaloFraction)),
    supportHalo: rounded(mean((sample) => sample.supportHaloFraction)),
    transportActivity: rounded(mean((sample) => sample.transportActivityFraction)),
    displacement: rounded(mean((sample) => sample.maximumDisplacementFineCells ?? 0), 2),
    solveIterations: rounded(mean((sample) => sample.executedSolveIterations), 2),
  };
};

const results = [];
for (const arm of arms) {
  console.error(`[losasso activity] running ${arm.id} (${arm.steps} advances)`);
  const runResult = await run(arm);
  const census = runResult.census;
  assert.equal(census.samples.length, arm.steps,
    `${arm.id} published ${census.samples.length} activity samples for ${arm.steps} advances`);
  const window = Math.max(1, Math.min(50, Math.floor(census.samples.length / 5)));
  results.push({
    arm: arm.id,
    gate: runResult.gate,
    invalidReceipts: census.samples.filter((sample) => !sample.receiptValid).length,
    invalidTransportReceipts: census.samples.filter((sample) => !sample.transportReceiptValid).length,
    invalidReceiptDetails: census.samples.filter((sample) => !sample.receiptValid).map((sample) => ({
      step: sample.step,
      generation: sample.generation,
      pageDeltaGeneration: sample.pageDeltaGeneration,
      live: sample.liveBandPages,
      active: sample.activeTransportPages,
      sleeping: sample.sleepingTransportPages,
    })),
    early: summarize(census.samples.slice(0, window)),
    late: summarize(census.samples.slice(-window)),
  });
}

console.table(results.flatMap((result) => [
  { arm: result.arm, gate: result.gate, window: "early", ...result.early },
  { arm: result.arm, gate: result.gate, window: "late", ...result.late },
]));
console.log(JSON.stringify({ phase: "losasso-activity-benchmark", results }));
if (results.some((result) => result.gate === "failed")) process.exitCode = 1;
