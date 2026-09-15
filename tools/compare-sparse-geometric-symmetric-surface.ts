/** Compare equal-step output from the adaptive and all-fine Dawn surface probes. */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { radialComparisonSvg, radialFrontsFromFields, radialOutlineSvg,
  type RadialFrontReceipt } from
  "./sparse-geometric-radial-front";

type Surface = { columns: number; missingColumns: number; mean_cells: number | null;
  rmsVariation_cells: number | null; symmetry: { maximumAbsoluteError_cells: number } };
type Checkpoint = { step: number; time_s: number; acceptedVolumeFine3: number;
  publishedPhi: { values: Array<number | null>; symmetry: { maximumAbsoluteError: number };
    upperSurface: Surface };
  density: number[]; densitySurface: Surface;
  phiVsDensityUpperSurface: { rms_cells: number | null; maximumAbsolute_cells: number;
    phaseMismatchColumns: number };
  radialFront?: { densityIntegrated: RadialFrontReceipt; phiGround: RadialFrontReceipt;
    phiProjected: RadialFrontReceipt };
  adaptivePhi: { fault: number; activeVertices: number; constrainedVertices: number } };
type Report = { arm: string; dimensions: [number, number, number]; cellSize_m: number;
  completed: boolean; failure?: string; checkpoints: Checkpoint[] };

const argument = (name: string) => process.argv.slice(2)
  .find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3);
const adaptivePath = argument("before") ?? argument("adaptive");
const benchmarkPath = argument("after") ?? argument("benchmark");
assert.ok(adaptivePath && benchmarkPath,
  "usage: --adaptive=<adaptive.json> --benchmark=<all-fine.json> [--output=<comparison.json>]");
const adaptive = JSON.parse(readFileSync(resolve(adaptivePath), "utf8")) as Report;
const benchmark = JSON.parse(readFileSync(resolve(benchmarkPath), "utf8")) as Report;
assert.deepEqual(adaptive.dimensions, benchmark.dimensions, "probe dimensions differ");
assert.equal(adaptive.cellSize_m, benchmark.cellSize_m, "probe cell sizes differ");

function fieldDifference(left: Array<number | null>, right: Array<number | null>) {
  assert.equal(left.length, right.length);
  let compared = 0, missingPhaseDisagreements = 0, sumSquared = 0, maximumAbsolute = 0;
  for (let i = 0; i < left.length; i++) {
    const a = left[i], b = right[i];
    if (a === null || b === null || !Number.isFinite(a) || !Number.isFinite(b)) {
      if ((a === null) !== (b === null)) missingPhaseDisagreements++;
      continue;
    }
    const error = a - b;
    compared++; sumSquared += error * error;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(error));
  }
  return { compared, missingPhaseDisagreements,
    rms: compared ? Math.sqrt(sumSquared / compared) : null, maximumAbsolute };
}

const benchmarkByStep = new Map(benchmark.checkpoints.map(row => [row.step, row]));
const requestedRadialSteps = (argument("steps") ?? "0,3,7,10").split(",").map(Number);
const radialEvolution = Object.fromEntries(([ ["before", adaptive],
  ["after", benchmark] ] as const).map(([run, report]) => [run,
  requestedRadialSteps.flatMap(step => {
    const checkpoint = report.checkpoints.find(row => row.step === step);
    if (!checkpoint) return [];
    const fronts = checkpoint.radialFront
      ?? radialFrontsFromFields(checkpoint.publishedPhi.values, checkpoint.density,
        report.dimensions);
    return [{ step, densityIntegrated: fronts.densityIntegrated,
      phiGround: fronts.phiGround, phiProjected: fronts.phiProjected }];
  })]));
const rows = adaptive.checkpoints.flatMap(observed => {
  const expected = benchmarkByStep.get(observed.step);
  if (!expected) return [];
  const phi = fieldDifference(observed.publishedPhi.values, expected.publishedPhi.values);
  const density = fieldDifference(observed.density, expected.density);
  const height = observed.publishedPhi.upperSurface;
  const expectedHeight = expected.publishedPhi.upperSurface;
  return [{ step: observed.step, time_s: observed.time_s,
    adaptive: { volumeFine3: observed.acceptedVolumeFine3,
      surfaceColumns: height.columns, surfaceMean_cells: height.mean_cells,
      surfaceVariation_cells: height.rmsVariation_cells,
      surfaceSymmetry_cells: height.symmetry.maximumAbsoluteError_cells,
      phiSymmetry: observed.publishedPhi.symmetry.maximumAbsoluteError,
      phiVsDensity: observed.phiVsDensityUpperSurface, adaptivePhi: observed.adaptivePhi },
    benchmark: { volumeFine3: expected.acceptedVolumeFine3,
      surfaceColumns: expectedHeight.columns, surfaceMean_cells: expectedHeight.mean_cells,
      surfaceVariation_cells: expectedHeight.rmsVariation_cells,
      surfaceSymmetry_cells: expectedHeight.symmetry.maximumAbsoluteError_cells,
      phiSymmetry: expected.publishedPhi.symmetry.maximumAbsoluteError,
      phiVsDensity: expected.phiVsDensityUpperSurface, adaptivePhi: expected.adaptivePhi },
    errorAgainstAllFine: { publishedPhiMetres: phi, density, surfaceMean_cells:
      height.mean_cells === null || expectedHeight.mean_cells === null ? null
        : height.mean_cells - expectedHeight.mean_cells,
      surfaceColumnCount: height.columns - expectedHeight.columns,
      acceptedVolumeFine3: observed.acceptedVolumeFine3 - expected.acceptedVolumeFine3 } }];
});
const comparison = { probe: "sparse-geometric-symmetric-surface-comparison",
  reference: "same 3-D discretization forced to the finest surface rung; each completed post-presentation checkpoint reports whether every then-active brick was finest; transient prephysics support topology is not forced, so this is a shared-method resolution-sensitivity control rather than an analytic oracle",
  adaptive: { path: resolve(adaptivePath), completed: adaptive.completed, failure: adaptive.failure,
    checkpointCount: adaptive.checkpoints.length },
  benchmark: { path: resolve(benchmarkPath), completed: benchmark.completed, failure: benchmark.failure,
    checkpointCount: benchmark.checkpoints.length },
  comparedSteps: rows.length, radialEvolution, rows };
const json = JSON.stringify(comparison, null, 2) + "\n";
const output = argument("output");
if (output) { const path = resolve(output); mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, json); console.log(path); }
else console.log(json);
const svgOutput = argument("svg");
const first = adaptive.checkpoints[0], last = adaptive.checkpoints.at(-1);
const referenceLast = benchmark.checkpoints.at(-1);
if (svgOutput && first?.radialFront && last?.radialFront && referenceLast?.radialFront) {
  const path = resolve(svgOutput); mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, radialOutlineSvg([
    { label: "adaptive initial", color: "#94a3b8",
      radii_cells: first.radialFront.densityIntegrated.radii_cells },
    { label: `adaptive step ${last.step}`, color: "#f97316",
      radii_cells: last.radialFront.densityIntegrated.radii_cells },
    { label: `finest-surface reference step ${referenceLast.step}`, color: "#22c55e",
      radii_cells: referenceLast.radialFront.densityIntegrated.radii_cells },
  ], "Sparse geometric birdseye density front"));
}
const comparisonSvgOutput = argument("comparison-svg");
if (comparisonSvgOutput) {
  const selected: Array<{ run: "before" | "after"; step: number;
    fronts: ReturnType<typeof radialFrontsFromFields> }> = [];
  for (const [run, report] of [["before", adaptive], ["after", benchmark]] as const) {
    for (const step of requestedRadialSteps) {
      const checkpoint = report.checkpoints.find(row => row.step === step);
      if (!checkpoint) continue;
      selected.push({ run, step, fronts: checkpoint.radialFront
        ?? radialFrontsFromFields(checkpoint.publishedPhi.values, checkpoint.density,
          report.dimensions) });
    }
  }
  const path = resolve(comparisonSvgOutput); mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, radialComparisonSvg(selected));
}
