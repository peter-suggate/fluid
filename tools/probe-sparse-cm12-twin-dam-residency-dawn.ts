/** Inspect Sparse CM12 brick residency after the twin-dam collision settles. */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createMinimalPowerDamBreak32Scene,
  createTwinDamCollisionScene,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
};
const sceneId = argument("scene") ?? "twin-dam";
if (sceneId !== "twin-dam" && sceneId !== "mini32") {
  throw new RangeError("scene must be twin-dam or mini32");
}
const seconds = Number(argument("seconds") ?? (sceneId === "mini32" ? 16 / 3 : 4));
const dt_s = 1 / 30;
const steps = Math.round(seconds / dt_s);

await acquireWebGPUExclusiveLock(
  "dawn-acceptance", "tools/probe-sparse-cm12-twin-dam-residency-dawn.ts",
);
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
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });
  try {
    const scene = sceneId === "mini32"
      ? createMinimalPowerDamBreak32Scene() : createTwinDamCollisionScene();
    const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
      timeStep: "paper", resolutionMode: "adaptive",
    });
    const solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
    try {
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * dt_s, []), true);
      }
      await device.queue.onSubmittedWorkDone();
      const { density } = await solver.readDiagnosticFields();
      const activity = await solver.readGPUActivityPolicy();
      const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
      const bricks = activity.bricks.map((brick) => {
        let maximumDensity = 0;
        let mass_cells = 0;
        for (let z = 0; z < 8; z += 1) for (let y = 0; y < 8; y += 1) {
          for (let x = 0; x < 8; x += 1) {
            const gx = 8 * brick.coordinate[0] + x;
            const gy = 8 * brick.coordinate[1] + y;
            const gz = 8 * brick.coordinate[2] + z;
            const rho = density[gx + dimensions[0] * (gy + dimensions[1] * gz)]!;
            maximumDensity = Math.max(maximumDensity, rho);
            mass_cells += rho;
          }
        }
        return { coordinate: brick.coordinate, active: brick.active,
          acceptedResolution: brick.acceptedResolution,
          plannedResolution: brick.plannedResolution,
          candidateResolution: brick.candidateResolution,
          candidateStatus: brick.candidateStatus,
          transferStatus: brick.transferStatus,
          faceTransferStatus: brick.faceTransferStatus,
          maximumDensity, mass_cells, reasons: brick.reasons,
          supportMask: brick.supportMask };
      });
      console.log(JSON.stringify({ scene: scene.sceneId, time_s: steps * dt_s, dimensions,
        activeBricks: bricks.filter(({ active }) => active),
        topBricks: bricks.filter(({ coordinate }) =>
          coordinate[1] >= Math.floor(dimensions[1] / 16)),
        ceilingBricks: bricks.filter(({ coordinate }) =>
          coordinate[1] + 1 === dimensions[1] / 8),
        validationErrors }, null, 2));
    } finally {
      solver.destroy();
    }
  } finally {
    device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
