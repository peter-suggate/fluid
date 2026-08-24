/**
 * What is actually in a captured pressure film?
 *
 * The journal's iteration records are cheap to read and already covered by the
 * Dawn tests. The *snapshots* are not: they stay on the device because the view
 * reads them there, so nothing in the normal path ever looks at their numbers.
 * This is the one place that does — it copies each captured field back and
 * reports its distribution over the live cells.
 *
 * That distribution is what the overlay's colour scale has to be built on. A
 * ramp reference picked by eye would be a claim about magnitudes nobody had
 * measured; the point of this probe is that the numbers in the view's
 * calibration comment can be traced back to a run.
 *
 * It also answers the question the film exists to answer, in text: how far does
 * each iteration actually move the residual, and where does it stop moving it.
 *
 * Run:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *   node --import tsx tools/probe-sparse-cm12-pressure-journal.ts --scene=mini32
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import {
  createMinimalPowerDamBreak32Scene,
  createSparseCM12LongDamBreakScene,
  createSymmetricExpansionScene,
} from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import {
  SPARSE_CM12_PRESSURE_JOURNAL_FIELDS,
  sparseCM12PressureJournalSnapshotOffset,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-journal";

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;
const sceneName = argument("scene", "mini32");
const frames = Number(argument("frames", "12"));
const iterations = Number(argument("iterations", "64"));
const buildScene = sceneName === "long-dam" ? createSparseCM12LongDamBreakScene
  : sceneName === "symmetric" ? createSymmetricExpansionScene
  : createMinimalPowerDamBreak32Scene;

/** Distribution of |value| over the cells that carry an unknown. */
function magnitudes(values: Float32Array, live: Float32Array, count: number) {
  const kept: number[] = [];
  let nonFinite = 0;
  let negative = 0;
  let minimum = Infinity;
  let signedMaximum = -Infinity;
  for (let cell = 0; cell < count; cell += 1) {
    if ((live[cell] ?? 0) < 0.5) continue;
    const value = values[cell] ?? 0;
    if (!Number.isFinite(value)) { nonFinite += 1; continue; }
    if (value < 0) negative += 1;
    minimum = Math.min(minimum, value);
    signedMaximum = Math.max(signedMaximum, value);
    kept.push(Math.abs(value));
  }
  if (kept.length === 0) return { cells: 0, nonFinite };
  kept.sort((a, b) => a - b);
  const at = (fraction: number) =>
    kept[Math.min(kept.length - 1, Math.floor(fraction * kept.length))]!;
  const sum = kept.reduce((total, value) => total + value, 0);
  return {
    cells: kept.length, nonFinite,
    // The ramp is logarithmic, so the spread between the median and the peak is
    // the number that decides how many decades it has to cover.
    median: at(0.5), p90: at(0.9), p99: at(0.99), maximum: kept[kept.length - 1]!,
    mean: sum / kept.length,
    // Whether the field is signed decides diverging versus sequential: half a
    // diverging ramp is wasted on a field that never goes negative.
    negativeFraction: negative / kept.length,
    minimum, signedMaximum,
    // A field that is mostly exact zeros is a converged one; reporting the
    // fraction separately keeps those zeros out of the percentiles above.
    zeroFraction: kept.filter((value) => value === 0).length / kept.length,
  };
}

await acquireWebGPUExclusiveLock("dawn-probe", "tools/probe-sparse-cm12-pressure-journal.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const scene = buildScene();
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    timeStep: "scene", pressureIterations: iterations,
  });
  const solver = await adaptiveMassMethod.createSolverAsync!(
    device, scene, "balanced", { ...values, pressureJournal: true },
    undefined, () => {}) as WebGPUAdaptiveMassSolver;
  const dt_s = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;

  // Film the last frame rather than the first: a t=0 solve starts from a
  // hydrostatic-free state and converges unrepresentatively fast.
  for (let frame = 1; frame < frames; frame += 1) {
    while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
  }
  assert.ok(solver.sparseWorldUI.control.pressureFilm!.setCaptureEnabled(true),
    "the solver reserved no journal");
  while (!solver.advanceTo(frames * dt_s, [])) await new Promise(setImmediate);

  const journal = await solver.sparseWorldUI.diagnostics.readPressureFilm();
  assert.ok(journal, "the armed advance left no capture");
  const source = solver.sparseWorldUI.overlays.pressureFilm;
  assert.ok(source, "the solver published no journal source");

  // One readback for every captured field of every captured iteration. This is
  // the expensive thing the live view exists to avoid; here it is the point.
  const cells = source.cellCount;
  const readback = device.createBuffer({
    label: "journal snapshot readback",
    size: 4 * cells,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const read = async (floatOffset: number): Promise<Float32Array> => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source.state, 4 * floatOffset, readback, 0, 4 * cells);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    return copy;
  };
  const live = await read(source.liquidFloatOffset);

  const snapshots: unknown[] = [];
  for (const [slot, iteration] of journal.snapshotIterations.entries()) {
    const fields: Record<string, unknown> = {};
    for (const [field, name] of SPARSE_CM12_PRESSURE_JOURNAL_FIELDS.entries()) {
      const offset = source.journalFloatOffset
        + sparseCM12PressureJournalSnapshotOffset(source.layout, slot, field);
      fields[name] = magnitudes(await read(offset), live, cells);
    }
    snapshots.push({ slot, iteration, fields });
  }
  readback.destroy();

  console.log(JSON.stringify({
    probe: "sparse-cm12-pressure-journal",
    scene: sceneName, frame: frames, cells,
    liveCells: live.reduce((total, value) => total + (value >= 0.5 ? 1 : 0), 0),
    encodedIterations: journal.encodedIterations,
    executedIterations: journal.executedIterations,
    firstCrossingIteration: journal.firstCrossingIteration,
    records: journal.records.map((record) => ({
      iteration: record.iteration, active: record.active,
      alpha: record.alpha, beta: record.beta, gamma: record.gamma,
      recursiveRelativeL2: record.recursiveRelativeL2,
      guardedRelativeL2: record.guardedRelativeL2,
      snapshot: record.snapshot,
    })),
    snapshots,
  }, null, 2));
  solver.destroy();
} finally {
  releaseWebGPUExclusiveLock();
}
// Dawn keeps handles that hold the loop open after the device is gone; the
// probe's whole output is already on stdout by here.
process.exit(0);
