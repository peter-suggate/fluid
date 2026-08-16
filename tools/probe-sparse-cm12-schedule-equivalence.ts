/**
 * Receipt for the topology scheduler: the exact per-frame decision sequence,
 * plus a density digest, so a reimplementation can be proved to choose the same
 * bricks rather than merely to run faster.
 *
 * Run in each arm's tree and diff the JSON.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const FRAMES = Number(process.argv.slice(2)
  .find((value) => value.startsWith("--frames="))?.slice(9) ?? 120);

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-sparse-cm12-schedule-equivalence.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });
  const scene = createSparseCM12LongDamBreakScene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", { timeStep: "scene" });
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
  const decisions: string[] = [];
  for (let frame = 1; frame <= FRAMES; frame += 1) {
    while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
    const info = await solver.readStats();
    decisions.push([
      info.adaptiveTopologyUrgentQueuedBrickCount,
      info.adaptiveTopologyOrdinaryQueuedBrickCount,
      info.adaptiveTopologyPreparedBrickCount,
      info.adaptiveTopologyCommittedBrickCount,
      info.adaptiveTopologyDeferredBrickCount,
      info.adaptiveAcceptedCellCount,
      info.adaptiveAcceptedRowCount,
      info.adaptiveFineBrickCount,
      info.adaptiveCoarseBrickCount,
    ].join(","));
  }
  const density = (await solver.readDiagnosticFields()).density;
  const digest = createHash("sha256").update(Buffer.from(density.buffer)).digest("hex");
  let front = -1;
  const [nx, ny, nz] = [192, 96, 32];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1)
    for (let x = 0; x < nx; x += 1)
      if (density[x + nx * (y + ny * z)]! >= 0.5) front = Math.max(front, x);
  console.log(JSON.stringify({
    probe: "sparse-cm12-schedule-equivalence", frames: FRAMES,
    decisionDigest: createHash("sha256").update(decisions.join("\n")).digest("hex"),
    densityDigest: digest, liquidFrontX: front,
    decisions, validationErrors,
  }, null, 2));
  solver.destroy();
} finally {
  releaseWebGPUExclusiveLock();
}
