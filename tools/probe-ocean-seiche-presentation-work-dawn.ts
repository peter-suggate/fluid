/**
 * Read the FPP1 work receipts for ocean-seiche under a full-domain min-8
 * policy. This is intentionally presentation-only evidence: it does not alter
 * solver construction or presentation policy.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-ocean-seiche-presentation-work-dawn.ts
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

async function bufferSha256(device: GPUDevice, source: GPUBuffer,
  size: number): Promise<string> {
  const readback = device.createBuffer({ size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return createHash("sha256").update(new Uint8Array(readback.getMappedRange())).digest("hex");
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

const input = createOceanSeicheScene();
input.fluid.refinementRegions = [{
  id: "presentation-work-domain-min8",
  rule: "minimum-cell-size",
  minimumCellSize_cells: 8,
  min_m: { x: -0.5 * input.container.width_m, y: 0,
    z: -0.5 * input.container.depth_m },
  max_m: { x: 0.5 * input.container.width_m, y: input.container.height_m,
    z: 0.5 * input.container.depth_m },
}];

await acquireWebGPUExclusiveLock("dawn-probe",
  "tools/probe-ocean-seiche-presentation-work-dawn.ts");
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
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene",
    resolutionMode: "adaptive",
    brickFineResolution: "8",
    presentationPageResolution: "8",
  });
  const solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
    device, input, "balanced", undefined, adaptiveMassSolverOptions(values), () => {});
  const dt_s = input.numerics.fixedDt_s ?? input.numerics.maxDt_s;
  try {
    const receipts: unknown[] = [];
    for (let step = 1; step <= steps; step += 1) {
      while (!solver.advanceTo(step * dt_s, [])) await new Promise(setImmediate);
      await device.queue.onSubmittedWorkDone();
      const [header, pages, stats] = await Promise.all([
        solver.readFramePlanPresentationHeaderQA(),
        solver.readPresentationPageAllocatorReceiptQA(),
        solver.readStats(),
      ]);
      receipts.push({ step, header, pages, work: {
        acceptedCells: stats.adaptiveAcceptedCellCount,
        acceptedRows: stats.adaptiveAcceptedRowCount,
        residentBricks: stats.adaptiveResidentBrickCount,
        topologyGeneration: stats.adaptiveTopologyShadowGeneration,
      } });
    }
    const presentation = solver.globalFineLevelSetSource;
    const terminalPresentation = {
      samplesBytes: presentation.plan.payloadCapacityBytes,
      samplesSha256: await bufferSha256(device, presentation.samples,
        presentation.plan.payloadCapacityBytes),
      metadataBytes: presentation.metadata.size,
      metadataSha256: await bufferSha256(device, presentation.metadata,
        presentation.metadata.size),
    };
    console.log(JSON.stringify({ steps, receipts, terminalPresentation,
      validationErrors }, null, 2));
    assert.deepEqual(validationErrors, []);
  } finally {
    solver.destroy();
  }
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
