/**
 * Focused bit-exact Sparse CM12 optimization oracle.
 *
 * Capture the correctness baseline:
 *   npm run oracle:sparse-cm12:bit-exact -- --write=/tmp/sparse-cm12-baseline.json
 *
 * Compare an optimized build against it:
 *   npm run oracle:sparse-cm12:bit-exact -- --compare=/tmp/sparse-cm12-baseline.json
 *
 * Establish that the current build/device is repeatable before optimizing:
 *   npm run oracle:sparse-cm12:bit-exact -- --repeat=2
 *
 * The tool takes the repository-wide Dawn lock before importing WebGPU. It
 * hashes raw Float32 bits; no tolerance, rounding, or summary statistic can
 * hide a changed value. Baselines are backend/adapter-specific and must be
 * captured and compared on the same device.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const SCHEMA = "fluid.sparse-cm12.bit-exact.v2";
// The canonical exactness capture spans two physical seconds. A caller may
// request a smaller development probe explicitly, but it is not a release
// oracle and must not replace the default receipt.
const DEFAULT_STEPS = 500;

interface HashReceipt {
  readonly elements: number;
  readonly bytes: number;
  readonly sha256: string;
}

interface OracleReceipt {
  readonly schema: typeof SCHEMA;
  readonly scene: "symmetric-expansion";
  readonly dimensions: readonly [number, number, number];
  readonly steps: number;
  readonly dt_s: number;
  readonly activityAcceptedSteps: number;
  readonly generation: number;
  readonly acceptedCellCount: number;
  readonly acceptedRowCount: number;
  readonly fields: {
    readonly density: HashReceipt;
    readonly velocity: HashReceipt;
    readonly pressure: HashReceipt;
    readonly divergence: HashReceipt;
  };
  readonly acceptedResolutions: HashReceipt & {
    readonly counts: Readonly<Record<"1" | "2" | "4" | "8" | "16", number>>;
  };
  readonly combinedSha256: string;
}

interface OracleRuntime {
  readonly createScene: () => ReturnType<
    typeof import("../lib/core/scenes")["createSymmetricExpansionScene"]
  >;
  readonly requiredLimits: typeof import("../lib/core/webgpu-device-limits")[
    "requiredFluidDeviceLimits"
  ];
  readonly Solver: typeof import(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver"
  )["WebGPUAdaptiveMassSolver"];
  readonly activityPolicy: typeof import(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident"
  )["SPARSE_CM12_ACTIVITY_POLICY"];
}

async function loadRuntime(sourceRoot: string): Promise<OracleRuntime> {
  const moduleUrl = (relative: string) => pathToFileURL(`${sourceRoot}/${relative}`).href;
  const [scenes, limits, solver, resident] = await Promise.all([
    import(moduleUrl("lib/core/scenes.ts")),
    import(moduleUrl("lib/core/webgpu-device-limits.ts")),
    import(moduleUrl("lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts")),
    import(moduleUrl("lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts")),
  ]);
  return {
    createScene: scenes.createSymmetricExpansionScene,
    requiredLimits: limits.requiredFluidDeviceLimits,
    Solver: solver.WebGPUAdaptiveMassSolver,
    activityPolicy: resident.SPARSE_CM12_ACTIVITY_POLICY,
  } as OracleRuntime;
}

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
};

const positiveInteger = (name: string, fallback: number): number => {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer; received ${value}`);
  }
  return value;
};

const bytesOf = (view: ArrayBufferView): Uint8Array =>
  new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

const sha256 = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex");

function hashReceipt(view: ArrayBufferView, elements: number): HashReceipt {
  return { elements, bytes: view.byteLength, sha256: sha256(bytesOf(view)) };
}

function compareReceipt(expected: OracleReceipt, actual: OracleReceipt): void {
  assert.equal(actual.schema, expected.schema, "oracle schema changed");
  assert.equal(actual.scene, expected.scene, "oracle scene changed");
  assert.deepEqual(actual.dimensions, expected.dimensions, "oracle dimensions changed");
  assert.equal(actual.steps, expected.steps, "oracle step count changed");
  assert.equal(actual.dt_s, expected.dt_s, "oracle timestep changed");
  for (const field of ["density", "velocity", "pressure", "divergence"] as const) {
    assert.deepEqual(actual.fields[field], expected.fields[field],
      `${field} raw-bit receipt changed`);
  }
  assert.equal(actual.combinedSha256, expected.combinedSha256,
    "combined Sparse CM12 physical-state receipt changed");
}

async function capture(
  device: GPUDevice,
  steps: number,
  runtime: OracleRuntime,
): Promise<OracleReceipt> {
  const scene = runtime.createScene();
  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
  let solver: Awaited<ReturnType<OracleRuntime["Solver"]["createAsync"]>> | undefined;
  try {
    solver = await runtime.Solver.createAsync(
      device,
      scene,
      "balanced",
      undefined,
      {
        resolutionMode: "adaptive",
        brickFineResolution: 16,
        presentationPageResolution: 16,
        surfaceFineRings: 1,
        receiverSupportRings: 3,
        receiverFloor: 1,
        timeStep: "scene",
        activityPolicy: runtime.activityPolicy,
      },
      () => {},
    );
    for (let step = 1; step <= steps; step += 1) {
      assert.equal(solver.advanceTo(step * dt_s, []), true,
        `step ${step} did not encode exactly once`);
    }
    await device.queue.onSubmittedWorkDone();
    const fields = await solver.readDiagnosticFields();
    const activity = await solver.readGPUActivityPolicy();
    const stats = await solver.readStats();
    const ordered = [...activity.bricks].sort((left, right) => left.key - right.key);
    // Hash key + accepted level together. A reordered or differently addressed
    // atlas cannot accidentally look equal because it has the same histogram.
    const resolutionWords = new Uint32Array(2 * ordered.length);
    const counts: Record<"1" | "2" | "4" | "8" | "16", number> = {
      "1": 0, "2": 0, "4": 0, "8": 0, "16": 0,
    };
    for (let index = 0; index < ordered.length; index += 1) {
      const brick = ordered[index]!;
      resolutionWords[2 * index] = brick.key;
      resolutionWords[2 * index + 1] = brick.acceptedResolution;
      counts[String(brick.acceptedResolution) as keyof typeof counts] += 1;
    }
    const fieldReceipts = {
      density: hashReceipt(fields.density, fields.density.length),
      velocity: hashReceipt(fields.velocity, fields.velocity.length),
      pressure: hashReceipt(fields.pressure, fields.pressure.length),
      divergence: hashReceipt(fields.divergence, fields.divergence.length),
    };
    const acceptedResolutions = {
      ...hashReceipt(resolutionWords, ordered.length),
      counts,
    };
    const combined = createHash("sha256");
    for (const [label, view] of [
      ["density", fields.density], ["velocity", fields.velocity],
      ["pressure", fields.pressure], ["divergence", fields.divergence],
    ] as const) {
      combined.update(label); combined.update("\0"); combined.update(bytesOf(view));
    }
    return {
      schema: SCHEMA,
      scene: "symmetric-expansion",
      dimensions: [solver.info.nx, solver.info.ny, solver.info.nz],
      steps,
      dt_s,
      activityAcceptedSteps: activity.acceptedSteps,
      generation: stats.fluidBrickGeneration ?? 0,
      acceptedCellCount: stats.adaptiveAcceptedCellCount ?? 0,
      acceptedRowCount: stats.adaptiveAcceptedRowCount ?? 0,
      fields: fieldReceipts,
      acceptedResolutions,
      combinedSha256: combined.digest("hex"),
    };
  } finally {
    solver?.destroy();
  }
}

const steps = positiveInteger("steps", DEFAULT_STEPS);
const repeat = positiveInteger("repeat", 1);
const writePath = argument("write");
const comparePath = argument("compare");
if (writePath && comparePath) throw new Error("choose either --write or --compare");
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const backend = argument("backend") ?? process.env.FLUID_WEBGPU_BACKEND
  ?? process.env.WEBGPU_BACKEND ?? "metal";
const sourceRoot = argument("source-root") ?? process.cwd();

await acquireWebGPUExclusiveLock(
  "dawn-acceptance",
  "tools/capture-sparse-cm12-bit-exact-dawn.ts",
);
let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${backend}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error(`No Dawn adapter is available for backend ${backend}`);
  const runtime = await loadRuntime(sourceRoot);
  device = await adapter.requestDevice({
    requiredLimits: runtime.requiredLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });
  const receipt = await capture(device, steps, runtime);
  for (let run = 1; run < repeat; run += 1) {
    compareReceipt(receipt, await capture(device, steps, runtime));
  }
  if (comparePath) {
    compareReceipt(JSON.parse(await readFile(comparePath, "utf8")) as OracleReceipt, receipt);
  }
  assert.deepEqual(validationErrors, [], "Dawn reported validation errors");
  if (writePath) await writeFile(writePath, `${JSON.stringify(receipt, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
