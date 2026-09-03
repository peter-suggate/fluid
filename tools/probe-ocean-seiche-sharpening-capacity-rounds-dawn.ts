/**
 * Finds the first fixed point of the conservative density-capacity relay on
 * ocean-seiche with a full-domain min8 region. Each arm advances normally up
 * to the sampled frame, then stops that frame after N of the maximum 8 relay
 * rounds. Consecutive bit-identical fields prove that the later round was an
 * identity for this state; once equal, every remaining round is also an
 * identity because the relay predicates see unchanged state.
 */
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
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
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

function scene() {
  const result = createOceanSeicheScene();
  result.fluid.refinementRegions = [{
    id: "sharpening-capacity-round-probe-domain-min8",
    rule: "minimum-cell-size",
    minimumCellSize_cells: 8,
    min_m: { x: -0.5 * result.container.width_m, y: 0,
      z: -0.5 * result.container.depth_m },
    max_m: { x: 0.5 * result.container.width_m, y: result.container.height_m,
      z: 0.5 * result.container.depth_m },
  }];
  return result;
}

async function runArm(device: GPUDevice, rounds: number) {
  const input = scene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene",
    resolutionMode: "adaptive",
    brickFineResolution: "8",
    presentationPageResolution: "8",
  });
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, input, "balanced", values, undefined, () => {},
  ) as WebGPUAdaptiveMassSolver;
  const dt_s = input.numerics.fixedDt_s ?? input.numerics.maxDt_s;
  try {
    for (let step = 1; step <= steps; step += 1) {
      if (step === steps) {
        solver.sparseWorldTrace.setSharpeningPhaseLimitForQA(
          `capacity-${rounds}` as Parameters<
            typeof solver.sparseWorldTrace.setSharpeningPhaseLimitForQA
          >[0],
        );
      }
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
      await device.queue.onSubmittedWorkDone();
    }
    const [fields, stats] = await Promise.all([
      solver.readDiagnosticFields(false, "candidate"),
      solver.readStats(),
    ]);
    let overCapacityCells = 0;
    let maximumExcess = 0;
    for (let cell = 0; cell < fields.density.length; cell += 1) {
      const excess = fields.density[cell]! - fields.solidOpenFraction[cell]!;
      if (excess > 0) {
        overCapacityCells += 1;
        maximumExcess = Math.max(maximumExcess, excess);
      }
    }
    return {
      rounds,
      density: sha256(fields.density),
      gamma: sha256(fields.gamma),
      sharpeningDelta: sha256(fields.sharpeningDelta),
      sharpeningReceiptMass: sha256(fields.sharpeningReceiptMass),
      overCapacityCells,
      maximumExcess,
      acceptedCells: stats.adaptiveAcceptedCellCount,
      acceptedRows: stats.adaptiveAcceptedRowCount,
    };
  } finally {
    solver.destroy();
  }
}

await acquireWebGPUExclusiveLock("dawn-probe",
  "tools/probe-ocean-seiche-sharpening-capacity-rounds-dawn.ts");
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
  const arms: Awaited<ReturnType<typeof runArm>>[] = [];
  for (let rounds = 1; rounds <= 8; rounds += 1) {
    arms.push(await runArm(device, rounds));
  }
  const comparisons = arms.slice(1).map((arm, index) => {
    const previous = arms[index]!;
    return {
      fromRounds: previous.rounds,
      toRounds: arm.rounds,
      densityEqual: previous.density === arm.density,
      gammaEqual: previous.gamma === arm.gamma,
      sharpeningDeltaEqual: previous.sharpeningDelta === arm.sharpeningDelta,
      sharpeningReceiptMassEqual:
        previous.sharpeningReceiptMass === arm.sharpeningReceiptMass,
    };
  });
  console.log(JSON.stringify({ steps, arms, comparisons, validationErrors }, null, 2));
  assert.deepEqual(validationErrors, []);
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
