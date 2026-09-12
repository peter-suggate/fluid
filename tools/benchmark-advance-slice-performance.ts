import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { advanceSlice, createAdvanceSlice, type AdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";

interface Options {
  readonly scenes: readonly string[];
  readonly warmup: number;
  readonly frames: number;
  readonly repetitions: number;
  readonly pressureIterations: number;
  readonly checkpoints: readonly number[];
  readonly output?: string;
  readonly compare?: string;
  readonly eagerDiagnostics: boolean;
  readonly pairedDiagnostics: boolean;
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.slice(2).find(value => value.startsWith(prefix))?.slice(prefix.length);
}

function integer(name: string, fallback: number): number {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Invalid --${name}`);
  return value;
}

const options: Options = {
  scenes: (argument("scenes") ??
    "coarse-first-pool-impact-half,twin-dam-collision,cm12-figure-3").split(","),
  warmup: integer("warmup", 2),
  frames: integer("frames", 4),
  repetitions: integer("repetitions", 5),
  pressureIterations: integer("pressure-iterations", 16),
  checkpoints: (argument("checkpoints") ?? "1,3,6").split(",").map(Number),
  output: argument("output"), compare: argument("compare"),
  eagerDiagnostics: process.argv.includes("--eager-diagnostics"),
  pairedDiagnostics: process.argv.includes("--paired-diagnostics"),
};

function advance(slice: AdvanceSlice, eagerDiagnostics = options.eagerDiagnostics): void {
  advanceSlice(slice, { pressureIterations: options.pressureIterations,
    eagerDiagnostics });
}

function digestBytes(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function mutableArrayDigests(slice: AdvanceSlice): Record<string, string> {
  const result: Record<string, string> = {};
  const seen = new WeakSet<object>();
  const visit = (value: unknown, path: string): void => {
    if (!value || typeof value !== "object" || seen.has(value as object)) return;
    seen.add(value as object);
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
      const view = value as Exclude<ArrayBufferView, DataView>;
      result[path] = `${view.constructor.name}:${view.byteLength}:` +
        digestBytes(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
      return;
    }
    if (value instanceof Map) {
      for (const [key, entry] of [...value.entries()].sort(([a], [b]) =>
        String(a).localeCompare(String(b)))) visit(entry, `${path}.map[${String(key)}]`);
      return;
    }
    if (value instanceof Set) return;
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i += 1) visit(value[i], `${path}[${i}]`);
      return;
    }
    for (const key of Object.keys(value as object).sort()) {
      if (path === "slice" && key === "scene") continue;
      visit((value as Record<string, unknown>)[key], `${path}.${key}`);
    }
  };
  visit(slice, "slice");
  return result;
}

function scalarState(slice: AdvanceSlice) {
  return {
    frame: slice.frame, time_s: slice.time_s, microsteps: slice.microsteps,
    maxVelocity: slice.maxVelocity, drift: slice.drift, churn: slice.churn,
    sourcePendingAreaFine: slice.sourcePendingAreaFine,
    iterations: slice.iterations, residual: slice.residual,
    seededVolume: slice.seededVolume, fault: slice.fault,
    topologyGeneration: slice.topology.accepted.generation,
    topologyCells: slice.numericalTopology.cells.map(cell => ({
      id: cell.id, stableId: cell.stableId, brickKey: cell.brickKey,
      minimum: cell.minimum, maximum: cell.maximum, widths: cell.widths,
    })),
    topologyRows: slice.numericalTopology.rows.map(row => ({
      id: row.id, kind: row.kind, axis: row.axis, center: row.center,
      area: row.area, distance: row.distance, openFraction: row.openFraction,
      openFractionBefore: row.openFractionBefore, openFractionAfter: row.openFractionAfter,
      solidVelocity: row.solidVelocity, terms: row.terms,
    })),
    pressureReceipt: slice.pressureReceipt,
    resolutionReceipt: slice.resolutionReceipt,
    sourceLedger: slice.sourceLedger,
    runtimeReceipt: slice.runtimeAuthority.receipt,
    scalarReceipt: slice.scalarAuthority.receipt,
    tracerReceipt: slice.tracerReceipt,
    retirementReceipt: slice.retirementAuthority.receipt,
    presentationReceipt: slice.presentation.receipt,
    rigidCouplingReceipts: slice.rigidCouplingReceipts,
    arrays: mutableArrayDigests(slice),
  };
}

function checkpoint(scene: string, frames: number) {
  const slice = createAdvanceSlice(productionSceneSliceSeedById(scene));
  for (let frame = 0; frame < frames; frame += 1)
    advance(slice);
  return scalarState(slice);
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function performanceCase(scene: string) {
  // One separate trajectory warms module/JIT paths. Every recorded repetition
  // starts from the same seed and advances the same state interval.
  const jit = createAdvanceSlice(productionSceneSliceSeedById(scene));
  for (let frame = 0; frame < options.warmup + options.frames; frame += 1)
    advance(jit);
  const elapsedMs: number[] = [];
  const endingState: string[] = [];
  let cells = 0, generation = 0;
  for (let repetition = 0; repetition < options.repetitions; repetition += 1) {
    const slice = createAdvanceSlice(productionSceneSliceSeedById(scene));
    for (let frame = 0; frame < options.warmup; frame += 1)
      advance(slice);
    const started = process.hrtime.bigint();
    for (let frame = 0; frame < options.frames; frame += 1)
      advance(slice);
    elapsedMs.push(Number(process.hrtime.bigint() - started) / 1e6);
    endingState.push(digestBytes(Buffer.from(JSON.stringify(scalarState(slice)))));
    cells = slice.numericalTopology.cells.length;
    generation = slice.topology.accepted.generation;
  }
  if (!endingState.every(hash => hash === endingState[0]))
    throw new Error(`${scene} is not deterministic across repetitions`);
  return { scene, dimensions: productionSceneSliceSeedById(scene).dimensions,
    warmupFrames: options.warmup, measuredFrames: options.frames,
    repetitions: options.repetitions, pressureIterations: options.pressureIterations,
    cells, generation, elapsedMs, medianElapsedMs: median(elapsedMs),
    medianFrameMs: median(elapsedMs) / Math.max(1, options.frames),
    minFrameMs: Math.min(...elapsedMs) / Math.max(1, options.frames),
    maxFrameMs: Math.max(...elapsedMs) / Math.max(1, options.frames),
    endingStateSha256: endingState[0] };
}

function pairedDiagnosticsCase(scene: string) {
  const elapsed = { eager: [] as number[], optimized: [] as number[] };
  const endingState = { eager: [] as string[], optimized: [] as string[] };
  for (const eager of [true, false]) {
    const jit = createAdvanceSlice(productionSceneSliceSeedById(scene));
    for (let frame = 0; frame < options.warmup + options.frames; frame += 1)
      advance(jit, eager);
  }
  let cells = 0, generation = 0;
  for (let repetition = 0; repetition < options.repetitions; repetition += 1) {
    // Alternate order to distribute thermal and background-load drift.
    for (const eager of repetition % 2 ? [false, true] : [true, false]) {
      const mode = eager ? "eager" : "optimized";
      const slice = createAdvanceSlice(productionSceneSliceSeedById(scene));
      for (let frame = 0; frame < options.warmup; frame += 1) advance(slice, eager);
      const started = process.hrtime.bigint();
      for (let frame = 0; frame < options.frames; frame += 1) advance(slice, eager);
      elapsed[mode].push(Number(process.hrtime.bigint() - started) / 1e6);
      endingState[mode].push(digestBytes(Buffer.from(JSON.stringify(scalarState(slice)))));
      cells = slice.numericalTopology.cells.length;
      generation = slice.topology.accepted.generation;
    }
  }
  const allHashes = [...endingState.eager, ...endingState.optimized];
  if (!allHashes.every(hash => hash === allHashes[0]))
    throw new Error(`${scene} diagnostic modes changed ending state`);
  const summarize = (values: readonly number[]) => ({ elapsedMs: values,
    medianElapsedMs: median(values), medianFrameMs: median(values) / Math.max(1, options.frames),
    minFrameMs: Math.min(...values) / Math.max(1, options.frames),
    maxFrameMs: Math.max(...values) / Math.max(1, options.frames) });
  return { scene, dimensions: productionSceneSliceSeedById(scene).dimensions,
    warmupFrames: options.warmup, measuredFrames: options.frames,
    repetitions: options.repetitions, pressureIterations: options.pressureIterations,
    cells, generation, eagerDiagnostics: summarize(elapsed.eager),
    optimized: summarize(elapsed.optimized), endingStateSha256: allHashes[0] };
}

const report = {
  format: "advance-slice-performance-v1",
  runtime: { node: process.version, platform: process.platform, arch: process.arch },
  configuration: options,
  performance: options.scenes.map(scene => options.pairedDiagnostics
    ? pairedDiagnosticsCase(scene) : performanceCase(scene)),
  exactState: Object.fromEntries(options.scenes.map(scene => [scene,
    Object.fromEntries(options.checkpoints.map(frames => [frames, checkpoint(scene, frames)]))])),
};

if (options.compare) {
  const baseline = JSON.parse(readFileSync(options.compare, "utf8"));
  const expected = baseline.exactState;
  // Compare the same persisted representation on both sides: optional
  // undefined receipt fields intentionally disappear from JSON artifacts.
  const observed = JSON.parse(JSON.stringify(report.exactState));
  if (!isDeepStrictEqual(observed, expected)) {
    for (const scene of options.scenes) for (const frames of options.checkpoints) {
      const before = expected?.[scene]?.[frames], after = observed?.[scene]?.[frames];
      if (!isDeepStrictEqual(before, after)) {
        const changedArrays = [...new Set([
          ...Object.keys(before?.arrays ?? {}), ...Object.keys(after?.arrays ?? {}),
        ])].filter(path => before?.arrays?.[path] !== after?.arrays?.[path]);
        console.error(JSON.stringify({ exactStateMismatch: { scene, frames, changedArrays,
          scalarStateChanged: !isDeepStrictEqual(
            { ...before, arrays: undefined }, { ...after, arrays: undefined }) } }, null, 2));
      }
    }
    process.exitCode = 1;
  }
}

const json = JSON.stringify(report, null, 2) + "\n";
if (options.output) writeFileSync(options.output, json);
else process.stdout.write(json);
