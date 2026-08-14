import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import type { PowerDamPerformanceSummary } from "./power-dam-performance-report";
import "../lib/methods";

const root = fileURLToPath(new URL("..", import.meta.url));
const benchmark = fileURLToPath(new URL("./benchmark-power-dam.ts", import.meta.url));
const dtValues = (process.argv.find((value) => value.startsWith("--dt="))
  ?.slice(5) ?? "0.004,0.002,0.001").split(",").map(Number);
const target_s = Number(process.argv.find((value) => value.startsWith("--target-s="))
  ?.slice(11) ?? 2);
const rounds = Number(process.argv.find((value) => value.startsWith("--rounds="))
  ?.slice(9) ?? 2);
if (dtValues.some((dt) => !Number.isFinite(dt) || dt <= 0)
  || !Number.isSafeInteger(rounds) || rounds < 1) {
  throw new RangeError("dt values must be positive and rounds must be a positive integer");
}

async function run(dt: number): Promise<PowerDamPerformanceSummary> {
  const child = spawn(process.execPath, ["--import", "tsx", benchmark,
    "--lane=large", `--dt=${dt}`, `--target-s=${target_s}`, "--json"], {
    cwd: root, env: process.env, stdio: ["ignore", "pipe", "inherit"],
  });
  let last = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    for (const line of chunk.split(/\r?\n/)) if (line.trim()) last = line.trim();
  });
  const code = await new Promise<number>((resolve, reject) => {
    child.once("error", reject); child.once("exit", (value) => resolve(value ?? 1));
  });
  if (code !== 0) throw new Error(`dt=${dt} benchmark exited ${code}`);
  return JSON.parse(last) as PowerDamPerformanceSummary;
}

const samples = new Map<number, number[]>();
for (let round = 0; round < rounds; round += 1) {
  const order = round % 2 === 0 ? dtValues : [...dtValues].reverse();
  for (const dt of order) {
    const summary = await run(dt);
    const values = samples.get(dt) ?? [];
    values.push(summary.advanceWall_ms); samples.set(dt, values);
    console.error(JSON.stringify({ phase: "dt-elasticity-sample", round, dt,
      steps: summary.steps, advanceWall_ms: summary.advanceWall_ms }));
  }
}
const median = (values: readonly number[]) => {
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.floor(ordered.length / 2)]!;
};
const points = dtValues.map((dt) => ({ dt, values_ms: samples.get(dt) ?? [],
  median_msPerAdvance: median(samples.get(dt) ?? []) }));
const first = points[0]!, last = points[points.length - 1]!;
const logElasticity = Math.log(last.median_msPerAdvance / first.median_msPerAdvance)
  / Math.log(last.dt / first.dt);
console.log(JSON.stringify({ schemaVersion: 1, experiment: "X-7-dt-elasticity",
  target_s, rounds, points, logElasticity,
  interpretation: Math.abs(logElasticity) < 0.2 ? "rebuild-dominated-flat"
    : logElasticity > 0.6 ? "change-scaled" : "mixed" }, null, 2));
