import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import "../lib/methods";

type TerminalCounters = Readonly<Record<string, number | undefined>>;
interface BenchmarkSummary {
  readonly scenario: string;
  readonly method: string;
  readonly steps: number;
  readonly advanceWall_ms: number;
  readonly validationErrorCount: number;
  readonly terminalCounters?: TerminalCounters;
}

const root = fileURLToPath(new URL("..", import.meta.url));
const benchmark = fileURLToPath(new URL("./benchmark-power-dam.ts", import.meta.url));
const rawSteps = process.argv.find((argument) => argument.startsWith("--steps="))
  ?.slice("--steps=".length) ?? "120";
const steps = Number(rawSteps);
if (!Number.isSafeInteger(steps) || steps < 12) {
  throw new RangeError(`--steps must be an integer of at least 12; received ${rawSteps}`);
}
const jsonOnly = process.argv.includes("--json");

const run = (reasonCones: boolean): Promise<BenchmarkSummary> => new Promise((resolve, reject) => {
  const child = spawn(process.execPath, ["--import", "tsx", benchmark,
    "--lane=large", `--steps=${steps}`, "--json"], {
    cwd: root,
    env: { ...process.env, FLUID_FINE_REASON_CONES: reasonCones ? "1" : "0" },
    stdio: ["ignore", "pipe", "inherit"],
  });
  let output = "";
  child.stdout!.on("data", (chunk: Buffer) => { output += chunk.toString(); });
  child.once("error", reject);
  child.once("exit", (code, signal) => {
    if (signal || code !== 0) {
      reject(new Error(`large-lane ${reasonCones ? "reason-cone" : "clean-control"}`
        + ` run failed (${signal ?? `exit ${code}`})`));
      return;
    }
    const line = output.trim().split("\n").findLast((candidate) => candidate.startsWith("{"));
    if (!line) { reject(new Error("benchmark produced no JSON summary")); return; }
    try { resolve(JSON.parse(line) as BenchmarkSummary); }
    catch (error) { reject(new Error(`invalid benchmark JSON: ${line}`, { cause: error })); }
  });
});

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2 : sorted[middle]!;
};
const correctnessSignature = (summary: BenchmarkSummary): string => JSON.stringify({
  scenario: summary.scenario, method: summary.method, steps: summary.steps,
  validationErrorCount: summary.validationErrorCount,
  terminalCounters: summary.terminalCounters,
});

// Interleaved A/B/A/B is the repository's clean-wall protocol. Both arms use
// normal GPU-resident scheduling and all unconditional topology, restriction,
// generation, capacity, publication, and pressure tripwires. The control is
// the prior broad-interface cone, not the rejected membership-only oracle.
const schedule = [false, true, false, true] as const;
const records: Array<{ arm: "clean-control" | "reason-cones"; summary: BenchmarkSummary }> = [];
for (const reasonCones of schedule) records.push({
  arm: reasonCones ? "reason-cones" : "clean-control",
  summary: await run(reasonCones),
});
const control = records.filter((record) => record.arm === "clean-control");
const candidate = records.filter((record) => record.arm === "reason-cones");
const referenceSignature = correctnessSignature(control[0]!.summary);
const exactControlParity = records.every((record) =>
  correctnessSignature(record.summary) === referenceSignature);
const allTripwiresPassed = records.every((record) =>
  record.summary.validationErrorCount === 0);
const controlWall_ms = median(control.map((record) => record.summary.advanceWall_ms));
const candidateWall_ms = median(candidate.map((record) => record.summary.advanceWall_ms));
const report = {
  schemaVersion: 1,
  authority: "Aanjaneya et al. 2017 Section 5 dynamic topology and complete narrow-band fast march",
  lane: "large", steps, schedule: schedule.map((value) => value ? "reason-cones" : "clean-control"),
  correctness: {
    passed: allTripwiresPassed && exactControlParity,
    allTripwiresPassed, exactTerminalParityWithCleanControl: exactControlParity,
    referenceTerminalCounters: control[0]!.summary.terminalCounters,
  },
  wall: {
    cleanControlMedian_msPerAdvance: controlWall_ms,
    reasonConesMedian_msPerAdvance: candidateWall_ms,
    delta_msPerAdvance: candidateWall_ms - controlWall_ms,
    deltaPercent: 100 * (candidateWall_ms - controlWall_ms) / controlWall_ms,
    samples: records.map((record) => ({ arm: record.arm,
      msPerAdvance: record.summary.advanceWall_ms })),
  },
};

if (jsonOnly) console.log(JSON.stringify(report));
else {
  console.log(`Power Liquids large lane (${steps} advances, A/B/A/B)`);
  console.log(`correctness: ${report.correctness.passed ? "PASS" : "FAIL"}`
    + ` · tripwires ${allTripwiresPassed ? "clear" : "FAILED"}`
    + ` · terminal parity ${exactControlParity ? "exact" : "DIFFERS"}`);
  console.log(`clean control: ${controlWall_ms.toFixed(3)} ms/advance`);
  console.log(`reason cones: ${candidateWall_ms.toFixed(3)} ms/advance`);
  console.log(`delta: ${report.wall.delta_msPerAdvance.toFixed(3)} ms/advance`
    + ` (${report.wall.deltaPercent.toFixed(2)}%)`);
  console.log(JSON.stringify(report));
}
if (!report.correctness.passed) process.exitCode = 1;
