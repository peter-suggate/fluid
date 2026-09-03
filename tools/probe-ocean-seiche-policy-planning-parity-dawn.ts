/** Stepwise parity probe for compact policy-tile planning and B1 closure elision. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createOceanSeicheScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;
const steps = Number(argument("steps", "12"));
if (!Number.isSafeInteger(steps) || steps < 1) {
  throw new RangeError("steps must be a positive integer");
}

const sha256 = (view: ArrayBufferView): string => createHash("sha256").update(
  new Uint8Array(view.buffer, view.byteOffset, view.byteLength),
).digest("hex");
const sha256Text = (value: unknown): string => createHash("sha256")
  .update(JSON.stringify(value)).digest("hex");

function scene() {
  const result = createOceanSeicheScene();
  result.fluid.refinementRegions = [{
    id: "policy-planning-parity-domain-min8",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 8,
    min_m: { x: -0.5 * result.container.width_m, y: 0,
      z: -0.5 * result.container.depth_m },
    max_m: { x: 0.5 * result.container.width_m, y: result.container.height_m,
      z: 0.5 * result.container.depth_m },
  }];
  return result;
}

async function runArm(device: GPUDevice, compact: boolean) {
  const input = scene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene", resolutionMode: "adaptive",
    brickFineResolution: "8", presentationPageResolution: "8",
  });
  const create = compact
    ? WebGPUAdaptiveMassSolver.createRefinementPolicyLeaderCompactionOracleForQA
    : WebGPUAdaptiveMassSolver.createRefinementPolicyFullLeafOracleForQA;
  const solver = await create.call(WebGPUAdaptiveMassSolver, device, input,
    "balanced", undefined, adaptiveMassSolverOptions(values), () => {});
  const dt_s = input.numerics.fixedDt_s ?? input.numerics.maxDt_s;
  const result = [];
  try {
    for (let step = 1; step <= steps; step += 1) {
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
      await device.queue.onSubmittedWorkDone();
      const [fields, activity, stats] = await Promise.all([
        solver.readDiagnosticFields(false),
        solver.sparseWorldTrace.readActivitySnapshot(true),
        solver.readStats(),
      ]);
      result.push({
        step,
        fields: Object.fromEntries(Object.entries(fields).map(([name, field]) =>
          [name, sha256(field)])),
        activity: sha256Text(activity),
        work: {
          acceptedCells: stats.adaptiveAcceptedCellCount,
          acceptedRows: stats.adaptiveAcceptedRowCount,
          pressureCells: stats.adaptivePressureCellCount,
          pressureRows: stats.adaptivePressureActiveRowCount,
          pressureResidual: stats.pressureRelativeResidual,
          topologyGeneration: stats.adaptiveTopologyShadowGeneration,
          committedBricks: stats.adaptiveTopologyCommittedBrickCount,
        },
      });
    }
    return result;
  } finally {
    solver.destroy();
  }
}

await acquireWebGPUExclusiveLock("dawn-probe",
  "tools/probe-ocean-seiche-policy-planning-parity-dawn.ts");
let device: GPUDevice | undefined;
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault(); validationErrors.push(event.error.message);
  });
  const baseline = await runArm(device, false);
  const candidate = await runArm(device, true);
  const comparison = baseline.map((expected, index) => {
    const actual = candidate[index]!;
    return {
      step: expected.step,
      fieldDifferences: Object.keys(expected.fields).filter((name) =>
        actual.fields[name] !== expected.fields[name]),
      activityDifferent: actual.activity !== expected.activity,
      workDifferences: Object.keys(expected.work).filter((name) =>
        actual.work[name as keyof typeof actual.work]
          !== expected.work[name as keyof typeof expected.work]),
    };
  });
  console.log(JSON.stringify({ steps, comparison, validationErrors }, null, 2));
  assert.deepEqual(validationErrors, []);
  assert.ok(comparison.every((entry) => entry.fieldDifferences.length === 0
    && !entry.activityDifferent && entry.workDifferences.length === 0));
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
