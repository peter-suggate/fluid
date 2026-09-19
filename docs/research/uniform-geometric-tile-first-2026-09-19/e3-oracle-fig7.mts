/**
 * E3 conservation oracle: the conservative volume transport on a live tile set.
 *
 * Four arms of the SAME scene, run sequentially (one solver resident at a time,
 * because four 128^3 Uniform Geometric solvers is ~2.9 GB):
 *
 *   dense    transportWorkMap=dense             -- the reference schedule
 *   dense2   transportWorkMap=dense             -- identical; the noise floor
 *   tiles    transportWorkMap=tiles (default)   -- the experiment
 *   starved  transportWorkMap=tiles, margin -8  -- the NEGATIVE control: the
 *            live set is starved to the seed tiles, below what the measured
 *            displacement requires, so the front should stall while the total
 *            stays conserved (the gather writes zero outside the set, so a
 *            short set mispositions liquid; it never creates or destroys it).
 *
 * `uvAddDonor` accumulates through a float compare-exchange loop, so the dense
 * schedule is not bit-reproducible against itself. `dense2` is therefore the
 * only honest denominator for `tiles`: a difference at or under the dense/dense
 * difference measured nothing.
 *
 * At each sampled step every arm's V field (volumeA, which is what the
 * diagnostics reduction reads at the end of a step) and gamma field (gammaA)
 * are captured and differenced against `dense`.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   FLUID_ORACLE_OUT=<json> node --import tsx <this file>
 *
 * Env: FLUID_ORACLE_SCENE (cm12-figure-7), FLUID_ORACLE_STEPS (65),
 * FLUID_ORACLE_SAMPLES ("10,25,40,60"), FLUID_ORACLE_ARMS, FLUID_ORACLE_OUT.
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "/Users/petersuggate/code/me/fluid/lib/core/method-contract";
import { getSceneDefinition } from "/Users/petersuggate/code/me/fluid/lib/core/scenes";
import { sceneDocument } from "/Users/petersuggate/code/me/fluid/lib/core/scene-definition";
import { uniformVolumeMethod } from "/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-volume-method";
import { requiredFluidDeviceLimits } from "/Users/petersuggate/code/me/fluid/lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "/Users/petersuggate/code/me/fluid/lib/harness/webgpu-smoke-isolation";

const SCENE = process.env.FLUID_ORACLE_SCENE ?? "cm12-figure-7";
const STEPS = Number(process.env.FLUID_ORACLE_STEPS ?? 65);
const SAMPLES = (process.env.FLUID_ORACLE_SAMPLES ?? "10,25,40,60")
  .split(",").map((value) => Number(value.trim())).filter((value) => value > 0);
const ARM_VALUES: Record<string, Record<string, unknown>> = {
  dense: { transportWorkMap: "dense" },
  dense2: { transportWorkMap: "dense" },
  tiles: { transportWorkMap: "tiles" },
  // A margin of -8 starves the live set to the seed tiles themselves, below
  // what the measured displacement requires. Not reachable from the panel.
  starved: { transportWorkMap: "tiles", transportReach: -8 },
};
const ARMS = (process.env.FLUID_ORACLE_ARMS ?? "dense,dense2,tiles,starved").split(",");

interface Access {
  volumeA: GPUTexture; gammaA: GPUTexture;
}

async function readTexture(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const row = Math.ceil(texture.width * 4 / 256) * 256;
  const layers = texture.height * texture.depthOrArrayLayers;
  const staging = device.createBuffer({
    size: row * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture },
      { buffer: staging, bytesPerRow: row, rowsPerImage: texture.height },
      [texture.width, texture.height, texture.depthOrArrayLayers]);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(staging.getMappedRange());
    const out = new Float32Array(texture.width * layers);
    for (let i = 0; i < layers; i += 1) {
      out.set(source.subarray(i * row / 4, i * row / 4 + texture.width), i * texture.width);
    }
    return out;
  } finally { staging.unmap(); staging.destroy(); }
}

function fieldSummary(values: Float32Array, nx: number, ny: number): {
  sum: number; nonzero: number; min: number[]; max: number[]; nonFinite: number;
} {
  let sum = 0, nonzero = 0, nonFinite = 0;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isFinite(value)) { nonFinite += 1; continue; }
    sum += value;
    if (Math.abs(value) > 1e-6) {
      nonzero += 1;
      const x = index % nx, y = Math.floor(index / nx) % ny, z = Math.floor(index / (nx * ny));
      const point = [x, y, z];
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis]!, point[axis]!);
        max[axis] = Math.max(max[axis]!, point[axis]!);
      }
    }
  }
  return { sum, nonzero, nonFinite,
    min: min.map((v) => Number.isFinite(v) ? v : -1), max: max.map((v) => Number.isFinite(v) ? v : -1) };
}

function difference(a: Float32Array, b: Float32Array, nx: number, ny: number) {
  let maxAbs = 0, sumAbs = 0, signedSum = 0, worst = -1;
  for (let index = 0; index < a.length; index += 1) {
    const delta = a[index]! - b[index]!;
    const absolute = Math.abs(delta);
    sumAbs += absolute; signedSum += delta;
    if (absolute > maxAbs) { maxAbs = absolute; worst = index; }
  }
  return {
    maxAbs: Number(maxAbs.toPrecision(6)),
    sumAbs: Number(sumAbs.toPrecision(6)),
    signedSum: Number(signedSum.toPrecision(6)),
    worstCell: worst < 0 ? null
      : [worst % nx, Math.floor(worst / nx) % ny, Math.floor(worst / (nx * ny))],
  };
}

await acquireWebGPUExclusiveLock("dawn-oracle", "docs/research/.../e3-oracle-fig7.mts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../../../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push((event as GPUUncapturedErrorEvent).error.message);
  });

  const captured = new Map<string, Map<number, { volume: Float32Array; gamma: Float32Array }>>();
  const armReports: Record<string, unknown>[] = [];
  let lattice = { nx: 0, ny: 0, nz: 0 };

  for (const arm of ARMS) {
    const scene = structuredClone(sceneDocument(getSceneDefinition(SCENE)));
    const values = resolveMethodValues(uniformVolumeMethod, "balanced",
      { timeStep: "scene", ...ARM_VALUES[arm] });
    const solver = await uniformVolumeMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {});
    const access = solver as unknown as Access;
    const dt_s = scene.numerics.maxDt_s;
    lattice = { nx: solver.info.nx, ny: solver.info.ny, nz: solver.info.nz };
    const fields = new Map<number, { volume: Float32Array; gamma: Float32Array }>();
    const series: Record<string, unknown>[] = [];
    const sampled: Record<string, unknown>[] = [];
    for (let frame = 1; frame <= STEPS; frame += 1) {
      while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
      const info = await solver.readStats() as unknown as Record<string, number | boolean>;
      series.push({
        frame,
        volumeCellSum: Number(Number(info.volumeCellSum ?? 0).toFixed(4)),
        maxSpeed_m_s: Number(Number(info.maxSpeed_m_s ?? 0).toFixed(4)),
        liveTiles: info.uniformTransportTiles ?? null,
        fineTiles: info.uniformTwoLevelFineTiles ?? null,
        displacement_cells: Number(Number(info.uniformTransportMaxDisplacement_cells ?? 0).toFixed(3)),
        requiredReach: info.uniformTransportRequiredReachTiles ?? null,
        configuredReach: info.uniformTransportReachTiles ?? null,
      });
      if (!SAMPLES.includes(frame)) continue;
      const volume = await readTexture(device, access.volumeA);
      const gamma = await readTexture(device, access.gammaA);
      fields.set(frame, { volume, gamma });
      sampled.push({
        frame,
        ...fieldSummary(volume, lattice.nx, lattice.ny),
        workMap: info.uniformTransportWorkMap ?? null,
        tilesTotal: info.uniformTransportTilesTotal ?? null,
        liveTiles: info.uniformTransportTiles ?? null,
        dustCells: info.uniformVolumeDustCells ?? null,
      });
    }
    captured.set(arm, fields);
    armReports.push({ arm, values: ARM_VALUES[arm], series, sampled });
    solver.destroy();
    await device.queue.onSubmittedWorkDone();
  }

  // Every arm is differenced against `dense`; `dense2` is the noise floor.
  const reference = captured.get("dense");
  const comparisons: Record<string, unknown>[] = [];
  if (reference) for (const arm of ARMS) {
    if (arm === "dense") continue;
    const other = captured.get(arm);
    if (!other) continue;
    for (const frame of SAMPLES) {
      const a = reference.get(frame), b = other.get(frame);
      if (!a || !b) continue;
      comparisons.push({
        arm, frame,
        volume: difference(b.volume, a.volume, lattice.nx, lattice.ny),
        gamma: difference(b.gamma, a.gamma, lattice.nx, lattice.ny),
      });
    }
  }

  const output = {
    phase: "uniform-geometric-e3-transport-live-set-oracle",
    capturedAt: new Date().toISOString(),
    scene: SCENE, steps: STEPS, samples: SAMPLES, lattice,
    adapter: (adapter as unknown as { info?: unknown }).info,
    arms: armReports, comparisons, validationErrors,
  };
  const out = process.env.FLUID_ORACLE_OUT;
  if (out) writeFileSync(out, JSON.stringify(output, null, 2));
  for (const report of armReports) {
    const arm = report.arm as string;
    const sampled = report.sampled as Record<string, number>[];
    for (const row of sampled) {
      console.log(`${arm.padEnd(7)} f${String(row.frame).padStart(3)}`
        + ` V=${row.sum!.toFixed(4)} cells=${row.nonzero} nonFinite=${row.nonFinite}`
        + ` live=${row.liveTiles ?? "-"}/${row.tilesTotal ?? "-"}`
        + ` extent=[${(row.min as unknown as number[]).join(",")}]..[${(row.max as unknown as number[]).join(",")}]`);
    }
  }
  for (const row of comparisons) {
    const volume = row.volume as Record<string, unknown>;
    const gamma = row.gamma as Record<string, unknown>;
    console.log(`diff ${String(row.arm).padEnd(7)} f${String(row.frame).padStart(3)}`
      + ` |dV|max=${volume.maxAbs} sum|dV|=${volume.sumAbs} signed=${volume.signedSum}`
      + ` |dGamma|max=${gamma.maxAbs}`);
  }
  console.log("validationErrors:", validationErrors);
  assert.deepEqual(validationErrors, []);
} finally {
  releaseWebGPUExclusiveLock();
}
