import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition, SPARSE_CM12_SYMMETRIC_EXPANSION_METHOD_PROFILE } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// Run one arm per process, under the same GPU lease as the regression suite.
// Compare the accepted volume averages, independently of the surface renderer.
const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "Set WEBGPU_NODE_MODULE to the native Dawn module path");
const maxCell = Number(process.env.SYMMETRIC_MAX_CELL ?? 0);
const steps = Number(process.env.SYMMETRIC_STEPS ?? 13);
const dt = Number(process.env.SYMMETRIC_DT ?? 1 / 30);
const output = process.env.SYMMETRIC_OUTPUT ?? "artifacts/symmetric-coarse-first-ab/coarse";
assert.ok([0, 1, 2, 4, 8].includes(maxCell));
assert.ok(Number.isSafeInteger(steps) && steps > 0);
assert.ok(Number.isFinite(dt) && dt > 0);
await acquireWebGPUExclusiveLock("dawn-probe", "symmetric-coarse-first-ab");
let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(modulePath).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const scene = sceneDocument(getSceneDefinition("sparse-cm12-symmetric-expansion"));
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  if (maxCell) scene.fluid.refinementRegions = [{
    id: "ab-whole-domain", rule: "minimum-cell-size", minimumCellSize_cells: 1,
    maximumCellSize_cells: maxCell,
    min_m: { x: -0.8, y: 0, z: -0.8 }, max_m: { x: 0.8, y: 0.8, z: 0.8 },
  }];
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    ...SPARSE_CM12_SYMMETRIC_EXPANSION_METHOD_PROFILE.overrides,
    timeStep: "scene", ...JSON.parse(process.env.SYMMETRIC_OVERRIDES ?? "{}"),
  });
  solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values,
    undefined, () => {}) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
  await mkdir(output, { recursive: true });
  const residentWGSLHash = createHash("sha256").update(await readFile(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url))).digest("hex");
  await writeFile(`${output}/configuration.json`, JSON.stringify({ scene, values, grid: [nx, ny, nz],
    steps, dt, maxCell, residentWGSLHash }, null, 2));
  const trace = [];
  for (let step = 0; step <= steps; step++) {
    if (step) {
      while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      assert.equal(solver.info.encodedSteps, step, "deferred preparation must not drop a step");
    }
    const fields = await solver.readDiagnosticFields(true);
    for (const name of ["density", "solidOpenFraction", "velocity", "pressure", "divergence"] as const) {
      const data = fields[name];
      assert.ok(data.every(Number.isFinite), `${name} finite at ${step}`);
      await writeFile(`${output}/${step}-${name}.bin`, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    }
    const activity = await solver.readGPUActivityPolicy();
    await writeFile(`${output}/${step}-activity.json`, JSON.stringify(activity));
    const stats = await solver.readStats();
    await writeFile(`${output}/${step}-stats.json`, JSON.stringify(stats));
    const heights = new Float32Array(nx * nz);
    let mass = 0, momentY = 0, momentR2 = 0, kinetic = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const at = x + nx * (y + ny * z), rho = fields.density[at]!;
      mass += rho; heights[x + nx * z] += rho;
      momentY += rho * (y + 0.5); momentR2 += rho * ((x + 0.5 - nx / 2) ** 2 + (z + 0.5 - nz / 2) ** 2);
      kinetic += 0.5 * rho * (fields.velocity[4 * at]! ** 2 + fields.velocity[4 * at + 1]! ** 2 + fields.velocity[4 * at + 2]! ** 2);
    }
    await writeFile(`${output}/${step}-height.bin`, new Uint8Array(heights.buffer));
    const row = { step, time: step * dt, mass, centerOfMassY: momentY / mass,
      rmsRadius: Math.sqrt(momentR2 / mass), kinetic,
      cells: activity.bricks.filter(b => b.active).reduce((n, b) => n + b.acceptedResolution ** 3, 0),
      histogram: Object.fromEntries([1, 2, 4, 8].map(r => [r, activity.bricks.filter(b => b.active && b.acceptedResolution === r).length])),
      pressureIterations: stats.pressureIterationsExecuted, pressureResidual: stats.pressureRelativeResidual };
    trace.push(row); console.log(JSON.stringify(row));
    assert.deepEqual(errors, []);
  }
  await writeFile(`${output}/trace.json`, JSON.stringify(trace, null, 2));
} finally {
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
}
