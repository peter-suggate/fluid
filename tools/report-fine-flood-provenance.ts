/**
 * Reports the reach the Section 5 closest-point flood actually used.
 *
 * The redistance ladder is sized before the frame from the authored band. This
 * runs a scene, reads the seed links the flood left resident, and reports the
 * ladder the observed hops would have needed. A large positive surplus means
 * whole descending passes gathered at a radius no sample used.
 *
 * The measurement perturbs nothing: no shader variant is compiled and the
 * accumulation pass runs after the stepping loop, outside any accepted advance.
 *
 *   node --import tsx tools/report-fine-flood-provenance.ts [--steps=N] [--scene=mini|dam]
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import { createMinimalPowerDamBreakScene, createLargePowerDamBreakScene } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { WebGPUFineFloodProvenance } from "../lib/webgpu-fine-flood-provenance";
import {
  summarizeFineFloodLadder,
  type FineFloodLadderSummary,
} from "../lib/fine-flood-provenance";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

interface FloodProvenanceProjection {
  readonly globalFineFloodProvenanceSource?: Readonly<{
    source: WebGPUFineLevelSetBrickSource;
    encodedStrides: readonly number[];
  }>;
}

const argument = (name: string, fallback: string): string => {
  const match = process.argv.slice(2).find((value) => value.startsWith(`--${name}=`));
  return match === undefined ? fallback : match.slice(name.length + 3);
};

const steps = Number(argument("steps", "8"));
assert.ok(Number.isInteger(steps) && steps >= 1 && steps <= 500,
  "--steps must be an integer from 1 to 500");
const sceneName = argument("scene", "mini");
assert.ok(sceneName === "mini" || sceneName === "large", "--scene must be mini or large");

const percent = (value: number) => `${(100 * value).toFixed(1)}%`;

function report(summary: FineFloodLadderSummary): void {
  const { plan } = summary;
  process.stdout.write(`\nFine closest-point flood provenance (${sceneName}, ${steps} steps)\n`);
  process.stdout.write(`${"".padEnd(62, "-")}\n`);
  process.stdout.write(`resident samples          ${summary.resident.toLocaleString()}\n`);
  process.stdout.write(`  resolved                ${summary.resolved.toLocaleString()}\n`);
  process.stdout.write(`  unresolved              ${summary.unresolved.toLocaleString()}\n`);
  process.stdout.write(`  self-seeded             ${summary.selfSeeded.toLocaleString()} (${percent(summary.resolved > 0 ? summary.selfSeeded / summary.resolved : 0)} of resolved)\n`);
  process.stdout.write(`\nencoded ladder            [${plan.strides.join(", ")}]\n`);
  process.stdout.write(`  descending passes       ${plan.descendingPasses} (reach ${plan.descendingReach} fine cells)\n`);
  process.stdout.write(`  collar repairs          ${plan.collarRepairPasses}\n`);
  process.stdout.write(`  total encoded reach     ${plan.encodedReach} fine cells\n`);
  process.stdout.write(`\ndeepest hop observed      ${summary.maximumAxisHop} fine cells\n`);
  process.stdout.write(`  encoded passes to cover ${summary.requiredLadderPasses} of ${plan.strides.length}\n`);
  process.stdout.write(`  trailing reach surplus  ${summary.surplusLadderPasses}\n`);
  process.stdout.write(`\nresolved by encoded pass\n`);
  summary.cumulativeResolvedShare.forEach((share, pass) => {
    if (pass > plan.strides.length) return;
    if (pass > 0 && summary.cumulativeResolvedShare[pass - 1] >= 1) return;
    const stride = pass === 0 ? "seed" : `s${plan.strides[pass - 1]}`;
    const bar = "#".repeat(Math.round(share * 40));
    process.stdout.write(`  ${String(pass).padStart(2)} ${stride.padEnd(5)} ${bar.padEnd(40)} ${percent(share)}\n`);
  });
  if (!summary.coveredByEncodedLadder) {
    process.stdout.write(`\nThe deepest hop exceeds the ladder's ${plan.encodedReach}-cell reach. For a warm publication\n`);
    process.stdout.write(`that is expected: topology carries and remaps the previous closest-point field, so a\n`);
    process.stdout.write(`link can be older and deeper than any single frame's flood could have built.\n`);
  } else if (summary.surplusLadderPasses > 0) {
    process.stdout.write(`\n${summary.surplusLadderPasses} trailing pass(es) reach further than anything observed needed. Collar repairs\n`);
    process.stdout.write(`close page-boundary turning routes a straight-line hop does not measure, so treat\n`);
    process.stdout.write(`this as an upper bound on what could be pruned.\n`);
  }
  if (summary.unresolved > 0) {
    process.stdout.write(`\n${summary.unresolved.toLocaleString()} resident samples ended the flood with no seed.\n`);
  }
  process.stdout.write("\n");
}

await acquireWebGPUExclusiveLock("tool", "tools/report-fine-flood-provenance.ts");
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
  assert.ok(adapter.features.has("subgroups"), "flood provenance requires subgroups");
  const device = await adapter.requestDevice({
    requiredFeatures: ["subgroups"],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const scene = sceneName === "mini" ? createMinimalPowerDamBreakScene() : createLargePowerDamBreakScene();
  const solver = await octreeMethod.createSolverAsync!(
    device, scene, "balanced",
    { ...octreeMethod.presetFor("balanced"), globalFineLevelSetFactor: "4", secondaryParticles: "off" },
    undefined, () => {},
  ) as GPUSolverInstance;
  await device.queue.onSubmittedWorkDone();

  for (let step = 1; step <= steps; step += 1) {
    const requestedTime_s = step * scene.numerics.fixedDt_s!;
    while (!solver.advanceTo(requestedTime_s, [])) {
      await new Promise((resolve) => setImmediate(resolve));
    }
  }
  await device.queue.onSubmittedWorkDone();

  const projection = (solver as unknown as { octreeProjection?: FloodProvenanceProjection }).octreeProjection;
  assert.ok(projection, "octree projection was not exposed");
  const provenance = projection.globalFineFloodProvenanceSource;
  assert.ok(provenance, "no fine generation has been redistanced yet; raise --steps");

  const measurement = new WebGPUFineFloodProvenance(
    device, provenance.source, provenance.encodedStrides);
  const encoder = device.createCommandEncoder({ label: "Fine flood provenance" });
  measurement.encode(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const readback = await measurement.read();
  assert.ok(readback, "flood provenance readback did not complete");

  report(summarizeFineFloodLadder({
    strides: provenance.encodedStrides,
    resident: readback.resident,
    histogram: readback.histogram,
  }));
  process.stdout.write(`resident pages            ${readback.residentPages.toLocaleString()}\n\n`);

  measurement.destroy();
  solver.destroy();
  assert.deepEqual(validationErrors, [],
    `WebGPU validation errors: ${validationErrors.join("; ")}`);
  device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
