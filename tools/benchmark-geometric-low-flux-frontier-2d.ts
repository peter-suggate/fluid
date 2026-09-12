/** CPU prototype measurement; these timings are not production GPU evidence. */
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { createAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import {
  solveStaticLowFluxDenseCopy2D,
  solveStaticLowFluxDensePingPong2D,
  solveStaticLowFluxFrontier2D,
  staticLowFluxProblem2D,
  type StaticLowFluxMode2D,
  type StaticLowFluxProblem2D,
  type StaticLowFluxResult2D,
} from "../lib/core/geometric-low-flux-frontier/reference-2d";

function argument(name: string, fallback: number): number {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find(value => value.startsWith(prefix))?.slice(prefix.length);
  const value = Number(raw ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`invalid --${name}`);
  return value;
}

const repetitions = argument("repetitions", 7);
const warmup = argument("warmup", 2);

function localizedChain(cellCount = 16_384, chainLength = 64): StaticLowFluxProblem2D {
  const volumes = new Array<number>(cellCount).fill(0.5);
  const capacities = new Array<number>(cellCount).fill(1);
  const first = cellCount - chainLength;
  // This amount needs forty synchronized factor generations to absorb the
  // terminal inflow across the 64-cell chain.
  for (let cell = first; cell < cellCount; cell += 1) volumes[cell] = 0.995;
  return staticLowFluxProblem2D({ volumes, capacities,
    faces: Array.from({ length: chainLength - 1 }, (_, offset) => ({
      negativeCell: first + offset, positiveCell: first + offset + 1, lowFlux: 0.2,
    })) });
}

function topologyDerivedSyntheticMixedRung(): StaticLowFluxProblem2D {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const topology = slice.numericalTopology;
  const seam = topology.subfaces.find(face => face.negativeCell >= 0 && face.positiveCell >= 0
    && topology.rows[face.rowId]!.kind === "mixed-seam");
  if (!seam) throw new Error("native fixture has no internal mixed-rung seam");
  const capacities = Float32Array.from(topology.cells,
    cell => Math.fround(slice.fields.capacity[cell.id]! * cell.area));
  const volumes = Float32Array.from(capacities, capacity => Math.fround(0.5 * capacity));
  volumes[seam.positiveCell] = Math.fround(0.95 * capacities[seam.positiveCell]!);
  const seamFlux = Math.fround(0.1 * Math.min(
    capacities[seam.negativeCell]!, capacities[seam.positiveCell]!));
  return {
    volumes, capacities,
    sourceRates: new Float32Array(topology.cells.length), dt: 0,
    faces: topology.subfaces.map(face => ({ negativeCell: face.negativeCell,
      positiveCell: face.positiveCell,
      lowFlux: face.id === seam.id ? seamFlux : 0 })),
    incidences: topology.cells.map(cell => {
      const entries: { face: number; negative: boolean }[] = [];
      const source = topology.subfaceIncidences?.[cell.id];
      if (source) for (const entry of source)
        entries.push({ face: entry.subfaceId, negative: entry.negative });
      else for (const face of topology.subfaces) {
        if (face.negativeCell === cell.id) entries.push({ face: face.id, negative: true });
        else if (face.positiveCell === cell.id) entries.push({ face: face.id, negative: false });
      }
      return entries;
    }),
  };
}

const solvers = {
  "dense-copy": solveStaticLowFluxDenseCopy2D,
  "dense-ping-pong": solveStaticLowFluxDensePingPong2D,
  "active-frontier": solveStaticLowFluxFrontier2D,
} as const;

function hash(result: StaticLowFluxResult2D): string {
  const digest = createHash("sha256");
  for (const values of [result.factors, result.lowFlux, result.lowStateVolume])
    digest.update(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  digest.update(JSON.stringify(result.passReceipts.map(receipt => ({
    invalidCount: receipt.invalidCount, firstInvalid: receipt.firstInvalid,
    changedCells: receipt.changedCells,
  }))));
  return digest.digest("hex");
}

function percentile(values: readonly number[], fraction: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))]!;
}

function measure(problem: StaticLowFluxProblem2D) {
  const elapsed = Object.fromEntries(Object.keys(solvers).map(mode => [mode, [] as number[]])) as
    Record<StaticLowFluxMode2D, number[]>;
  for (let index = 0; index < warmup; index += 1)
    for (const solver of Object.values(solvers)) solver(problem);
  let results: Record<StaticLowFluxMode2D, StaticLowFluxResult2D> | undefined;
  for (let repetition = 0; repetition < repetitions; repetition += 1) {
    const order = repetition % 2
      ? ["active-frontier", "dense-ping-pong", "dense-copy"] as const
      : ["dense-copy", "dense-ping-pong", "active-frontier"] as const;
    const current = {} as Record<StaticLowFluxMode2D, StaticLowFluxResult2D>;
    for (const mode of order) {
      const started = performance.now();
      current[mode] = solvers[mode](problem);
      elapsed[mode].push(performance.now() - started);
    }
    results = current;
  }
  const baselineHash = hash(results!["dense-copy"]);
  for (const mode of Object.keys(solvers) as StaticLowFluxMode2D[]) {
    if (hash(results![mode]) !== baselineHash) throw new Error(`${mode} differs from dense-copy`);
  }
  return { cells: problem.volumes.length, faces: problem.faces.length,
    passes: results!["dense-copy"].passes, converged: results!["dense-copy"].converged,
    exactStateSha256: baselineHash,
    variants: Object.fromEntries((Object.keys(solvers) as StaticLowFluxMode2D[]).map(mode => {
      const result = results![mode];
      return [mode, { medianMs: percentile(elapsed[mode], 0.5),
        p95Ms: percentile(elapsed[mode], 0.95), samplesMs: elapsed[mode],
        updateCellVisits: result.updateCellVisits,
        bankCopyCellVisits: result.bankCopyCellVisits,
        frontierCommitCellVisits: result.frontierCommitCellVisits,
        totalWorkCellVisits: result.updateCellVisits + result.bankCopyCellVisits
          + result.frontierCommitCellVisits,
        peakFrontier: result.peakFrontier,
        peakFrontierAfterSeed: result.peakFrontierAfterSeed,
        evaluatedCellsPerPass: result.passReceipts.map(pass => pass.evaluatedCells.length),
        invalidCellsPerPass: result.passReceipts.map(pass => pass.invalidCount) }];
    })) };
}

const cases = [
  ["localized-chain-16384-64", localizedChain()] as const,
  ["topology-derived-synthetic-coarse-surface-translation-b8-b4",
    topologyDerivedSyntheticMixedRung()] as const,
];

console.log(JSON.stringify({ format: "geometric-low-flux-frontier-2d-v1",
  qualification: "CPU reference proof only; no production or GPU speedup claim",
  runtime: { node: process.version, platform: process.platform, arch: process.arch },
  warmup, repetitions,
  cases: Object.fromEntries(cases.map(([name, problem]) => [name, measure(problem)])),
}, null, 2));
