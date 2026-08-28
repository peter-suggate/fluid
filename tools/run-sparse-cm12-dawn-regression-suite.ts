/**
 * Fast, encompassing native-Dawn confidence gate for production Sparse CM12.
 *
 *   npm run test:dawn:sparse-cm12
 *   npm run test:dawn:sparse-cm12 -- --list
 *   npm run test:dawn:sparse-cm12 -- --lane=mini32-performance
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  readWebGPUExclusiveLockHolder,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  SPARSE_CM12_DAWN_LANES,
  SPARSE_CM12_DAWN_SUITE_BUDGET_MS,
  type SparseCM12DawnLane,
  type SparseCM12DawnPerformanceLane,
} from "./sparse-cm12-dawn-regression-manifest";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const NODE = process.execPath;
const DEFAULT_DAWN_MODULE = join(ROOT, "node_modules/webgpu/index.js");

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`--${name} requires a value`);
  return value;
};

const hasFlag = (name: string): boolean => process.argv.includes(`--${name}`);

interface ChildResult {
  readonly exitCode: number | null;
  readonly signal: NodeJS.Signals | null;
  readonly timedOut: boolean;
  readonly elapsedMs: number;
  readonly stdout: string;
  readonly stderr: string;
}

async function removeDeadChildLock(childPid: number | undefined): Promise<void> {
  const holder = await readWebGPUExclusiveLockHolder();
  if (childPid !== undefined && holder?.owner?.pid === childPid && !holder.alive) {
    await releaseWebGPUExclusiveLock();
  }
}

async function runChild(
  argv: readonly string[],
  environment: Readonly<Record<string, string>>,
  timeoutMs: number,
  forwardOutput: boolean,
): Promise<ChildResult> {
  const started = performance.now();
  const child = spawn(NODE, argv, {
    cwd: ROOT,
    env: {
      ...process.env,
      WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE,
      FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
      ...environment,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "", stderr = "", timedOut = false;
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    stdout += chunk;
    if (forwardOutput) process.stdout.write(chunk);
  });
  child.stderr.on("data", (chunk: string) => {
    stderr += chunk;
    if (forwardOutput) process.stderr.write(chunk);
  });
  const timer = setTimeout(() => {
    timedOut = true;
    child.kill("SIGTERM");
    setTimeout(() => child.kill("SIGKILL"), 2_000).unref();
  }, timeoutMs);
  let exitCode: number | null;
  let signal: NodeJS.Signals | null;
  try {
    ({ exitCode, signal } = await new Promise<{
      exitCode: number | null;
      signal: NodeJS.Signals | null;
    }>((resolve, reject) => {
      child.once("error", reject);
      child.once("exit", (code, exitSignal) => resolve({
        exitCode: code, signal: exitSignal,
      }));
    }));
  } finally {
    clearTimeout(timer);
    await removeDeadChildLock(child.pid);
  }
  return {
    exitCode, signal, timedOut, stdout, stderr,
    elapsedMs: performance.now() - started,
  };
}

function performanceArguments(
  lane: SparseCM12DawnPerformanceLane,
  outputPath: string,
): readonly string[] {
  return [
    "--import", "tsx", "tools/probe-sparse-cm12-stage-cost.ts",
    `--scene=${lane.scene}`,
    `--brick-fine=${lane.brickFineResolution}`,
    `--presentation-page=${lane.presentationPageResolution}`,
    `--warmup=${lane.warmupFrames}`,
    `--frames=${lane.measuredFrames}`,
    `--capture-gap-ms=${lane.captureGapMs}`,
    "--final-qa=0",
    "--enforce-pressure-receipts=0",
    "--quiet=1",
    `--out=${outputPath}`,
  ];
}

interface LaneReceipt {
  readonly id: string;
  readonly kind: SparseCM12DawnLane["kind"];
  readonly passed: boolean;
  readonly elapsedMs: number;
  readonly failure?: string;
  readonly performance?: {
    readonly medianAdvanceMs: number;
    readonly referenceMedianAdvanceMs: number;
    readonly maximumMedianAdvanceMs: number;
    readonly measuredFrames: number;
  };
}

async function runLane(lane: SparseCM12DawnLane, temporaryDirectory: string,
  timeoutMs = lane.timeoutMs):
Promise<LaneReceipt> {
  process.stdout.write(`\n[dawn-regression] ${lane.id}: ${lane.description}\n`);
  if (lane.kind === "correctness") {
    const result = await runChild([
      ...(lane.nodeOptions ?? []), "--import", "tsx", lane.testFile,
    ], lane.environment ?? {}, timeoutMs, true);
    const failure = result.timedOut
      ? `exceeded ${timeoutMs} ms timeout`
      : result.exitCode !== 0
        ? `exited ${result.exitCode ?? result.signal ?? "without status"}`
        : undefined;
    return { id: lane.id, kind: lane.kind, passed: failure === undefined,
      elapsedMs: result.elapsedMs, ...(failure ? { failure } : {}) };
  }

  const outputPath = join(temporaryDirectory, `${lane.id}.json`);
  const result = await runChild(performanceArguments(lane, outputPath), {},
    timeoutMs, false);
  if (result.timedOut || result.exitCode !== 0) {
    if (result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);
    return {
      id: lane.id, kind: lane.kind, passed: false, elapsedMs: result.elapsedMs,
      failure: result.timedOut ? `exceeded ${timeoutMs} ms timeout`
        : `probe exited ${result.exitCode ?? result.signal ?? "without status"}`,
    };
  }
  try {
    const report = JSON.parse(await readFile(outputPath, "utf8")) as {
      samples?: number;
      medianAdvance_ms?: number;
      diagnostic?: { passed?: boolean };
      validationErrors?: unknown[];
    };
    assert.equal(report.samples, lane.measuredFrames,
      "performance probe did not capture every requested frame");
    assert.equal(report.diagnostic?.passed, true,
      "performance probe diagnostics did not pass");
    assert.deepEqual(report.validationErrors, [],
      "performance probe reported Dawn validation errors");
    assert.ok(Number.isFinite(report.medianAdvance_ms),
      "performance probe omitted medianAdvance_ms");
    assert.ok(report.medianAdvance_ms! <= lane.maximumMedianAdvanceMs,
      `${lane.id} median ${report.medianAdvance_ms} ms exceeds ${
        lane.maximumMedianAdvanceMs} ms baseline ceiling`);
    process.stdout.write(`[dawn-regression] ${lane.id}: ${
      report.medianAdvance_ms!.toFixed(3)} ms median (ceiling ${
      lane.maximumMedianAdvanceMs.toFixed(3)} ms)\n`);
    return {
      id: lane.id, kind: lane.kind, passed: true, elapsedMs: result.elapsedMs,
      performance: {
        medianAdvanceMs: report.medianAdvance_ms!,
        referenceMedianAdvanceMs: lane.referenceMedianAdvanceMs,
        maximumMedianAdvanceMs: lane.maximumMedianAdvanceMs,
        measuredFrames: lane.measuredFrames,
      },
    };
  } catch (error) {
    return {
      id: lane.id, kind: lane.kind, passed: false, elapsedMs: result.elapsedMs,
      failure: error instanceof Error ? error.message : String(error),
    };
  }
}

function printHelp(): void {
  console.log(`Sparse CM12 Dawn regression suite

Usage:
  npm run test:dawn:sparse-cm12
  npm run test:dawn:sparse-cm12 -- --list
  npm run test:dawn:sparse-cm12 -- --lane=mini32-performance
  npm run test:dawn:sparse-cm12 -- --kind=correctness

Options:
  --list                 Print the canonical matrix without running Dawn
  --lane=ID              Run one named lane
  --kind=KIND            Run correctness or performance lanes
  --out=PATH             Write the combined JSON receipt
  --help, -h             Print this help

The full matrix has a ${SPARSE_CM12_DAWN_SUITE_BUDGET_MS / 1000}-second wall budget.`);
}

const requestedLane = argument("lane");
const requestedKind = argument("kind");
if (requestedKind !== undefined && requestedKind !== "correctness"
  && requestedKind !== "performance") {
  throw new Error("--kind must be correctness or performance");
}
if (requestedLane !== undefined
  && !SPARSE_CM12_DAWN_LANES.some((lane) => lane.id === requestedLane)) {
  throw new Error(`unknown lane ${requestedLane}`);
}
const selected = SPARSE_CM12_DAWN_LANES.filter((lane) =>
  (requestedLane === undefined || lane.id === requestedLane)
  && (requestedKind === undefined || lane.kind === requestedKind));

if (hasFlag("help") || process.argv.includes("-h")) {
  printHelp();
} else if (hasFlag("list")) {
  console.log(JSON.stringify({
    suite: "sparse-cm12-dawn-regression",
    budgetMs: SPARSE_CM12_DAWN_SUITE_BUDGET_MS,
    lanes: selected,
  }, null, 2));
} else {
  const existingHolder = await readWebGPUExclusiveLockHolder();
  if (existingHolder) throw new Error(
    `Cannot start Sparse CM12 Dawn regression while ${existingHolder.description} holds `
      + "the repository-wide WebGPU lease.",
  );
  const temporaryDirectory = await mkdtemp(join(tmpdir(), "fluid-cm12-regression-"));
  const startedAt = new Date().toISOString();
  const started = performance.now();
  const fullSuite = requestedLane === undefined && requestedKind === undefined;
  const receipts: LaneReceipt[] = [];
  try {
    for (const lane of selected) {
      const remainingMs = fullSuite
        ? SPARSE_CM12_DAWN_SUITE_BUDGET_MS - (performance.now() - started)
        : lane.timeoutMs;
      if (remainingMs <= 0) {
        receipts.push({ id: lane.id, kind: lane.kind, passed: false, elapsedMs: 0,
          failure: "suite exhausted its wall-clock budget before this lane" });
        continue;
      }
      receipts.push(await runLane(lane, temporaryDirectory,
        Math.min(lane.timeoutMs, remainingMs)));
    }
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
  const elapsedMs = performance.now() - started;
  const withinBudget = !fullSuite || elapsedMs <= SPARSE_CM12_DAWN_SUITE_BUDGET_MS;
  const report = {
    suite: "sparse-cm12-dawn-regression",
    startedAt,
    backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    elapsedMs: Number(elapsedMs.toFixed(1)),
    budgetMs: SPARSE_CM12_DAWN_SUITE_BUDGET_MS,
    withinBudget,
    passed: withinBudget && receipts.every((receipt) => receipt.passed),
    receipts,
  };
  const outputPath = argument("out");
  if (outputPath) await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`\n${JSON.stringify(report, null, 2)}`);
  if (!withinBudget) process.stderr.write(
    `[dawn-regression] suite exceeded ${SPARSE_CM12_DAWN_SUITE_BUDGET_MS} ms budget\n`);
  if (!report.passed) process.exitCode = 1;
}
