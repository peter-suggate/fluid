#!/usr/bin/env node
/** Paired five-frame dam64 wall-clock experiments for face/scalar transport. */
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const RUNNER = join(ROOT, "tools/run-sparse-cm12-temporal-regressions.ts");
const DAWN = process.env.WEBGPU_NODE_MODULE ?? join(ROOT, "node_modules/webgpu/index.js");
const BACKEND = process.env.FLUID_WEBGPU_BACKEND ?? "metal";

const argument = (name: string, fallback?: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length)
    ?? fallback;
};
const replays = Number(argument("replays", "3"));
const gap = Number(argument("gap-ms", "110"));
if (!Number.isSafeInteger(replays) || replays < 3) throw new RangeError("replays must be >=3");
if (!Number.isSafeInteger(gap) || gap < 100) throw new RangeError("gap-ms must be >=100");

const arms = [
  { name: "baseline", experiment: "baseline", cadence: 1 },
  { name: "legacy-owner-hash", experiment: "legacy-owner-hash", cadence: 1 },
  { name: "logical-owner-directory", experiment: "logical-owner-directory", cadence: 1 },
  { name: "logical-owner-mass-rung", experiment: "logical-owner-mass-rung", cadence: 1 },
  { name: "face-characteristic-cache", experiment: "face-characteristic-cache", cadence: 1 },
  { name: "face-row-packets", experiment: "face-row-packets", cadence: 1 },
  { name: "mass-rung-packets", experiment: "mass-rung-packets", cadence: 1 },
  { name: "mass-local-atomics", experiment: "mass-local-atomics", cadence: 1 },
  { name: "mass-swept-clean", experiment: "mass-swept-clean", cadence: 1 },
  { name: "structure-gamma-legacy", experiment: "structure-gamma-legacy", cadence: 1 },
  { name: "structure-mass-legacy", experiment: "structure-mass-legacy", cadence: 1 },
  { name: "structure-cache-legacy", experiment: "structure-cache-legacy", cadence: 1 },
  { name: "topology-cadence-2", experiment: "baseline", cadence: 2 },
  { name: "topology-cadence-4", experiment: "baseline", cadence: 4 },
  { name: "face-packets-cache", experiment: "face-packets-cache", cadence: 1 },
  { name: "mass-rung-local", experiment: "mass-rung-local", cadence: 1 },
  { name: "face-packets-mass-rung", experiment: "face-packets-mass-rung", cadence: 1 },
  { name: "all-valid", experiment: "all-valid", cadence: 1 },
] as const;
const requestedArms = new Set((argument("arms") ?? arms.map((arm) => arm.name).join(","))
  .split(",").filter(Boolean));
const selectedArms = arms.filter((arm) => arm.name === "baseline"
  || requestedArms.has(arm.name));
for (const requested of requestedArms) if (!arms.some((arm) => arm.name === requested)) {
  throw new RangeError(`unknown arm ${requested}`);
}

const run = (args: readonly string[]): Promise<void> => new Promise((resolve, reject) => {
  const child = spawn(process.execPath, ["--import", "tsx", RUNNER, ...args], {
    cwd: ROOT,
    env: { ...process.env, WEBGPU_NODE_MODULE: DAWN, FLUID_WEBGPU_BACKEND: BACKEND },
    stdio: ["ignore", "ignore", "inherit"],
  });
  child.once("error", reject);
  child.once("exit", (code, signal) => code === 0 ? resolve()
    : reject(new Error(`experiment child exited ${code ?? signal}`)));
});

type Distribution = { readonly median_ms: number; readonly p95_ms: number;
  readonly perStep: readonly { readonly median_ms: number }[] };
type Receipt = { readonly passed: boolean; readonly nonPressure: Distribution;
  readonly stages: Readonly<Record<string, Distribution>>;
  readonly physicsReference: unknown; readonly failures: readonly string[] };
const ratio = (value: number, baseline: number) => value / baseline;

const temporary = await mkdtemp(join(tmpdir(), "cm12-transport-experiments-"));
try {
  const receipts: Array<{ arm: typeof arms[number]; receipt: Receipt }> = [];
  for (const arm of selectedArms) {
    const path = join(temporary, `${arm.name}.json`);
    process.stderr.write(`\n[cm12 experiment] ${arm.name}\n`);
    await run(["--internal-dam-front-performance", `--performance-replays=${replays}`,
      `--performance-capture-gap-ms=${gap}`, `--transport-experiment=${arm.experiment}`,
      `--topology-cadence=${arm.cadence}`, `--record-performance-baseline=${path}`,
      "--allow-failing-experiment-record"]);
    receipts.push({ arm, receipt: JSON.parse(await readFile(path, "utf8")) as Receipt });
  }
  const baseline = receipts[0]!.receipt;
  const faceBaseline = baseline.stages["receiver-topology"]!;
  const massBaseline = baseline.stages["scalar-transport"]!;
  const result = {
    kind: "sparse-cm12-early-transport-experiments",
    version: 1,
    scene: "minimal-power-dam-break-64",
    frames: 5,
    replays,
    backend: BACKEND,
    arms: receipts.map(({ arm, receipt }) => {
      const physicsBitExactToBaseline = JSON.stringify(receipt.physicsReference)
        === JSON.stringify(baseline.physicsReference);
      return ({
      ...arm,
      passed: receipt.passed && physicsBitExactToBaseline,
      lanePassed: receipt.passed,
      physicsBitExactToBaseline,
      failures: receipt.failures,
      nonPressure: { median_ms: receipt.nonPressure.median_ms,
        p95_ms: receipt.nonPressure.p95_ms,
        medianRatio: ratio(receipt.nonPressure.median_ms, baseline.nonPressure.median_ms),
        perStepMedian_ms: receipt.nonPressure.perStep.map((step) => step.median_ms) },
      facePreparation: { median_ms: receipt.stages["receiver-topology"]!.median_ms,
        p95_ms: receipt.stages["receiver-topology"]!.p95_ms,
        medianRatio: ratio(receipt.stages["receiver-topology"]!.median_ms,
          faceBaseline.median_ms),
        perStepMedian_ms: receipt.stages["receiver-topology"]!.perStep.map(
          (step) => step.median_ms) },
      massTransport: { median_ms: receipt.stages["scalar-transport"]!.median_ms,
        p95_ms: receipt.stages["scalar-transport"]!.p95_ms,
        medianRatio: ratio(receipt.stages["scalar-transport"]!.median_ms,
          massBaseline.median_ms),
        perStepMedian_ms: receipt.stages["scalar-transport"]!.perStep.map(
          (step) => step.median_ms) },
    });}),
  };
  const json = `${JSON.stringify(result, null, 2)}\n`;
  const out = argument("out");
  if (out) await writeFile(out, json);
  process.stdout.write(json);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
