/**
 * Interleaved A/A + A/B driver for the power-dam throughput lanes.
 *
 * `benchmark-power-dam.ts` measures one configuration per invocation. Scoring a
 * cutover means comparing two, and the two traps that make a naive comparison
 * worthless are both structural:
 *
 *  1. **Arms must be separate processes.** Dawn on this build cannot safely
 *     rebuild an identical pipeline specialization inside one process -- the
 *     mini gate asserts exactly that (`benchmark-mini-dam-fluid-gate.ts`). So an
 *     in-process A/A is not merely noisy, it is not expressible.
 *  2. **Arms must be interleaved, never blocked.** Running all of A then all of
 *     B attributes thermal drift and background load to the flag under test.
 *     Round-robin ordering makes drift common-mode.
 *
 * And one measurement trap that is not structural but has burned this program
 * before: a difference is only real if it clears the noise floor *measured on
 * this machine, in this session*. So the control arm is always sampled twice per
 * round under two labels. That A/A pair costs one extra run per round and is the
 * only thing that licenses reading an A/B delta at all -- it is the protocol
 * `POWER_LIQUIDS_SCENARIO_LEAP_PLAN.md` §5 asks for ("A/A before A/B").
 *
 * Usage:
 *
 *   node --import tsx tools/benchmark-power-dam-ab.ts \
 *     --lane=large --steps=60 --repeats=3 \
 *     --arm=indirect:FLUID_COARSE_SUMMARY_INDIRECT_DISPATCH=1
 *
 * The control arm is implicit (the lane's own defaults) and needs no `--arm`.
 * Each `--arm=label:KEY=VALUE[,KEY=VALUE]` adds one variant. Every run goes
 * through `benchmark-power-dam.ts`, so the lane table, the tripwire floor and
 * the exclusive GPU lock are inherited rather than reimplemented here.
 */

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { POWER_DAM_LANE_ENVIRONMENT, type PowerDamRuntimeLane }
  from "./power-dam-lane-environment";
import "../lib/methods";

const root = fileURLToPath(new URL("..", import.meta.url));
const benchmark = fileURLToPath(new URL("./benchmark-power-dam.ts", import.meta.url));

const argv = process.argv.slice(2);
const flag = (name: string): string | undefined =>
  argv.find((entry) => entry.startsWith(`--${name}=`))?.slice(name.length + 3);

const lane = (flag("lane") ?? "mini") as PowerDamRuntimeLane;
if (!(lane in POWER_DAM_LANE_ENVIRONMENT)) {
  throw new Error(`--lane must be one of ${Object.keys(POWER_DAM_LANE_ENVIRONMENT).join(", ")}`);
}
const steps = flag("steps");
const repeats = Number(flag("repeats") ?? 3);
if (!Number.isInteger(repeats) || repeats < 2) {
  throw new Error("--repeats must be an integer >= 2; a single round cannot separate a delta from noise");
}

type Arm = { readonly label: string; readonly environment: Record<string, string> };

const variants: Arm[] = argv
  .filter((entry) => entry.startsWith("--arm="))
  .map((entry) => {
    const body = entry.slice("--arm=".length);
    const separator = body.indexOf(":");
    if (separator <= 0) {
      throw new Error(`--arm must be label:KEY=VALUE[,KEY=VALUE]; received "${body}"`);
    }
    const label = body.slice(0, separator);
    const environment: Record<string, string> = {};
    for (const assignment of body.slice(separator + 1).split(",").filter(Boolean)) {
      const equals = assignment.indexOf("=");
      if (equals <= 0) throw new Error(`arm "${label}" has a malformed assignment "${assignment}"`);
      environment[assignment.slice(0, equals)] = assignment.slice(equals + 1);
    }
    if (Object.keys(environment).length === 0) {
      throw new Error(`arm "${label}" sets no variables; it would duplicate the control`);
    }
    if (environment.FLUID_TRIPWIRES !== undefined) {
      // The benchmark floors tripwires at "1" regardless, so accepting this
      // would silently produce an arm that does not differ the way it claims.
      throw new Error(`arm "${label}" may not set FLUID_TRIPWIRES; detection is not an A/B axis`);
    }
    return { label, environment };
  });
if (variants.length === 0) {
  throw new Error("at least one --arm=label:KEY=VALUE is required; with none this is just a benchmark run");
}

/** The control, sampled twice per round under two labels. The pair is the noise
 * floor: same configuration, same round, different process. */
const CONTROL = "control";
const CONTROL_AA = "control-aa";
const arms: Arm[] = [
  { label: CONTROL, environment: {} },
  { label: CONTROL_AA, environment: {} },
  ...variants,
];
const duplicate = arms.find((arm, index) => arms.findIndex((other) => other.label === arm.label) !== index);
if (duplicate) throw new Error(`duplicate arm label "${duplicate.label}"`);

type Sample = { readonly round: number; readonly advanceWall_ms: number };
const samples = new Map<string, Sample[]>(
  [...arms.map((arm) => arm.label), "warmup"].map((label) => [label, [] as Sample[]]));
const failures: string[] = [];

/** One measured run. Everything about the scene, the tripwire floor and the GPU
 * lock lives in the child; this only chooses the arm's variables. */
const runArm = async (arm: Arm, round: number): Promise<void> => {
  const child = spawn(process.execPath, [
    "--import", "tsx", benchmark,
    `--lane=${lane}`,
    ...(steps === undefined ? [] : [`--steps=${steps}`]),
    "--json",
  ], {
    cwd: root,
    env: { ...process.env, ...arm.environment },
    stdio: ["ignore", "pipe", "inherit"],
  });
  let advanceWall_ms: number | undefined;
  let tripwireMode: string | undefined;
  const lines = createInterface({ input: child.stdout! });
  lines.on("line", (line) => {
    try {
      const record = JSON.parse(line) as
        { advanceWall_ms?: number; phase?: string; mode?: string };
      if (record.phase === "tripwires") tripwireMode = record.mode;
      if (typeof record.advanceWall_ms === "number") advanceWall_ms = record.advanceWall_ms;
    } catch { /* the child also prints non-JSON progress; ignore it */ }
  });
  const exitCode = await new Promise<number>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => signal
      ? reject(new Error(`Dawn benchmark exited from ${signal}`))
      : resolve(code ?? 1));
  });
  if (exitCode !== 0 || advanceWall_ms === undefined) {
    // A failed arm is reported, never averaged around: a lane that trips a
    // tripwire is not slower or faster, it is invalid.
    failures.push(`${arm.label} round ${round}: exit ${exitCode}${
      advanceWall_ms === undefined ? " with no advanceWall_ms" : ""}`);
    return;
  }
  if (tripwireMode === "failfast") {
    // +26.8% on the large lane, and it cannot be compared to anything measured
    // without the per-step fence. Refuse rather than publish the number.
    throw new Error(`arm ${arm.label} ran under FLUID_TRIPWIRES=failfast;`
      + " its per-step fence removes host/GPU overlap and inflates the wall."
      + " Unset FLUID_TRIPWIRES to measure.");
  }
  samples.get(arm.label)!.push({ round, advanceWall_ms });
  console.log(`  round ${round} · ${arm.label}: ${advanceWall_ms.toFixed(2)} ms/advance`);
};

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
};

console.log(`lane ${lane}${steps === undefined ? "" : ` · ${steps} steps`} · ${repeats} rounds`
  + ` · ${arms.length} arms (${arms.length * repeats} runs + 1 discarded warmup, interleaved)`);

// The first run of a session pays shader compilation and pipeline creation that
// no later run repeats. Measured: 337.13 ms against ~288 for every subsequent
// run on `large`, which alone pushed the A/A spread to 50.97 ms and the noise
// floor to 54.75 -- wide enough to swallow any real effect. It is one sample and
// it is not the population being measured, so it is spent, not recorded.
console.log("  warmup (discarded)");
await runArm({ label: "warmup", environment: {} }, 0);
samples.set("warmup", []);
failures.length = 0;

for (let round = 1; round <= repeats; round += 1) {
  // Round-robin, one child at a time: two Dawn clients on this GPU silently
  // change physics, and blocked ordering would alias thermal drift onto the
  // flag under test.
  for (const arm of arms) await runArm(arm, round);
}

const statistics = (label: string) => {
  const walls = samples.get(label)!.map((sample) => sample.advanceWall_ms);
  return walls.length === 0 ? undefined : {
    runs: walls.length,
    median_ms: median(walls),
    min_ms: Math.min(...walls),
    spread_ms: Math.max(...walls) - Math.min(...walls),
  };
};

const control = statistics(CONTROL);
const controlAA = statistics(CONTROL_AA);
if (!control || !controlAA) {
  console.error("\ncontrol arms produced no samples; nothing can be concluded");
  for (const entry of failures) console.error(`  failed: ${entry}`);
  process.exit(1);
}

/** What "no difference" looks like on this machine right now: the A/A gap plus
 * the larger of the two control spreads. A variant inside this band is not a
 * result, in either direction. */
const noiseFloor_ms = Math.abs(control.median_ms - controlAA.median_ms)
  + Math.max(control.spread_ms, controlAA.spread_ms);
const controlMedian_ms = median([...samples.get(CONTROL)!, ...samples.get(CONTROL_AA)!]
  .map((sample) => sample.advanceWall_ms));

console.log(`\nA/A noise floor: ${noiseFloor_ms.toFixed(2)} ms`
  + ` (|${control.median_ms.toFixed(2)} - ${controlAA.median_ms.toFixed(2)}|`
  + ` + worst spread ${Math.max(control.spread_ms, controlAA.spread_ms).toFixed(2)})`);
console.log(`control median: ${controlMedian_ms.toFixed(2)} ms/advance over ${
  control.runs + controlAA.runs} runs\n`);

const verdicts = variants.map((arm) => {
  const measured = statistics(arm.label);
  if (!measured) return { label: arm.label, verdict: "NO DATA" as const };
  const delta_ms = measured.median_ms - controlMedian_ms;
  const conclusive = Math.abs(delta_ms) > noiseFloor_ms;
  console.log(`${arm.label}: ${measured.median_ms.toFixed(2)} ms/advance`
    + ` (${delta_ms >= 0 ? "+" : ""}${delta_ms.toFixed(2)} ms,`
    + ` ${delta_ms >= 0 ? "+" : ""}${(100 * delta_ms / controlMedian_ms).toFixed(1)}%)`
    + ` · ${conclusive ? (delta_ms < 0 ? "FASTER" : "SLOWER") : "INCONCLUSIVE — inside the A/A floor"}`
    + ` · min ${measured.min_ms.toFixed(2)}, spread ${measured.spread_ms.toFixed(2)}`);
  return { label: arm.label, delta_ms, conclusive,
    verdict: conclusive ? (delta_ms < 0 ? "FASTER" : "SLOWER") : "INCONCLUSIVE" } as const;
});

if (failures.length !== 0) {
  console.error("\nfailed runs (excluded from every statistic above):");
  for (const entry of failures) console.error(`  ${entry}`);
}

console.log(`\n${JSON.stringify({ record: "power-dam-ab", lane, steps: steps ?? "lane default",
  repeats, noiseFloor_ms: Number(noiseFloor_ms.toFixed(3)),
  controlMedian_ms: Number(controlMedian_ms.toFixed(3)),
  arms: Object.fromEntries(arms.map((arm) => [arm.label, statistics(arm.label)])),
  verdicts, failures })}`);

// A failed run means the comparison rests on fewer samples than requested, and
// silently exiting 0 would let a half-measured A/B be quoted as a result.
if (failures.length !== 0) process.exit(1);
