/** Native receipts plus SIMD-Wasm UI-surface analysis for gentle-moving-blob. */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { createAdvanceView, type AdvanceGraph, type AdvanceView } from "../lib/physics-wasm/advance-view";
import { decodePhysicsPublication } from "../lib/physics-wasm/publication";
import { parsePhysicsReceipt } from "../lib/physics-wasm/protocol";
import { contourMetrics } from "./gentle-moving-blob-metrics";
import { loadFluidWasmForNode } from "./wasm/load-module.mjs";

type Arm = "lsv-moving" | "lsv-stationary" | "baseline-moving" | "lsv-pinned-fine";
type NativeFrame = Record<string, unknown> & { receipt: Record<string, unknown> };
type NativeOutput = { frames: NativeFrame[]; failure: unknown };
type WasmWorld = {
  advance(sequence: number, dt: number): string;
  receipt(): string;
  snapshot(mask: number): Uint8Array;
  free(): void;
};

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const argument = (name: string, fallback: string): string => {
  const prefix = `--${name}=`;
  return process.argv.find(value => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const requestedArm = argument("arm", "all");
const arms: Arm[] = requestedArm === "all"
  ? ["lsv-moving", "lsv-stationary", "baseline-moving", "lsv-pinned-fine"]
  : [requestedArm as Arm];
assert.ok(arms.every(arm => ["lsv-moving", "lsv-stationary", "baseline-moving", "lsv-pinned-fine"].includes(arm)),
  "--arm must be all, lsv-moving, lsv-stationary, baseline-moving, or lsv-pinned-fine");
const frames = Number(argument("frames", "180"));
assert.ok(Number.isSafeInteger(frames) && frames > 0, "--frames must be a positive integer");
const outputPath = argument("output", "artifacts/level-set-volume/gentle-moving-blob-analysis.json");
const binary = resolve(root, argument("binary", "rust/target/release/examples/verify_world"));
const dt = 1 / 30;
const h = 0.05;
const speed = 0.08;
const authoredRadius = 0.3;
// The even-depth centre slice samples the z=+h/2 voxel-centre plane.
const sliceZ = h / 2;
const sectionRadius = Math.sqrt(authoredRadius ** 2 - sliceZ ** 2);

function configuredScene(arm: Arm) {
  const definition = findSceneDefinition("gentle-moving-blob");
  assert.ok(definition, "gentle-moving-blob scene is missing");
  const document = structuredClone(sceneDocument(definition));
  if (arm === "lsv-stationary") document.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  if (arm === "lsv-pinned-fine") {
    document.fluid.refinementRegions = [{
      id: "gentle-moving-blob-uniform-fine",
      rule: "minimum-cell-size",
      minimumCellSize_cells: 1,
      maximumCellSize_cells: 1,
      min_m: { x: -0.8, y: 0, z: -0.4 },
      max_m: { x: 0.8, y: 1.2, z: 0.4 },
    }];
  }
  return { definition, document };
}

function nativeRun(arm: Arm, document: ReturnType<typeof sceneDocument>): NativeOutput {
  const transportExperiment = arm === "baseline-moving" ? "baseline" : "level-set-volume";
  const run = spawnSync(binary, {
    cwd: root,
    input: JSON.stringify({
      scene: document,
      productionOptions: { dtS: dt, timeStep: "paper" },
      worldOptions: {
        pressureIterations: 256,
        pressureRelativeTolerance: 1e-6,
        transportExperiment,
      },
      frames,
      receiptsOnly: true,
      captureFailure: true,
    }),
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
  if (run.status !== 0) throw new Error(run.stderr || run.stdout);
  return JSON.parse(run.stdout) as NativeOutput;
}

function eigenAxisRatio(xx: number, xy: number, yy: number): number | null {
  const trace = xx + yy;
  const d = Math.hypot(xx - yy, 2 * xy);
  const minor = 0.5 * (trace - d), major = 0.5 * (trace + d);
  return minor > 0 ? Math.sqrt(major / minor) : null;
}

function surfaceMetrics(view: AdvanceView, expected: [number, number]) {
  const segments = view.rdf.segmentsFine;
  assert.ok(view.rdf.receipt, "RDF receipt is missing");
  const representedAreaFine = Number(view.rdf.receipt.representedAreaFine);
  const exactAreaFine = Number(view.rdf.receipt.exactAreaFine);
  const contour = contourMetrics(segments, representedAreaFine, expected, sectionRadius / h,
    h, [view.nx, view.ny]);
  const phi = view.rdf.vertexPhiFine;
  let unresolved = 0, symmetricDifferenceCells = 0, outsideCells = 0, missingCells = 0;
  for (let y = 0; y < view.ny; y++) for (let x = 0; x < view.nx; x++) {
    const stride = view.nx + 1;
    const values = [phi[x + stride * y], phi[x + 1 + stride * y],
      phi[x + stride * (y + 1)], phi[x + 1 + stride * (y + 1)]];
    if (values.some(value => !Number.isFinite(value))) { unresolved++; continue; }
    const rdfLiquid = values.reduce((sum, value) => sum + value!, 0) <= 0;
    const analyticLiquid = Math.hypot(x + 0.5 - expected[0], y + 0.5 - expected[1]) <= sectionRadius / h;
    if (rdfLiquid !== analyticLiquid) symmetricDifferenceCells++;
    if (rdfLiquid && !analyticLiquid) outsideCells++;
    if (!rdfLiquid && analyticLiquid) missingCells++;
  }
  return {
    ...contour,
    representedAreaM2: representedAreaFine * h * h,
    referenceVolumeAreaM2: exactAreaFine * h * h,
    rdfRasterSymmetricDifferenceM2: symmetricDifferenceCells * h * h,
    rdfRasterOutsideAnalyticM2: outsideCells * h * h,
    rdfRasterMissingAnalyticM2: missingCells * h * h,
    rdfRasterUnresolvedCells: unresolved,
  };
}

function covarianceAxisRatio(value: unknown): number | null {
  const c = value as [[number, number], [number, number]] | undefined;
  return c ? eigenAxisRatio(c[0][0], c[0][1], c[1][1]) : null;
}

async function wasmRun(arm: Arm, document: ReturnType<typeof sceneDocument>, native: NativeOutput) {
  const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
  const transportExperiment = arm === "baseline-moving" ? "baseline" : "level-set-volume";
  const scene = { id: "gentle-moving-blob", label: `Gentle moving blob (${arm})`, document };
  const world = wasm.FluidWorld.from_scene(JSON.stringify(document), JSON.stringify({
    runEpoch: 1,
    commandSequence: 0,
    pressureIterations: 256,
    pressureRelativeTolerance: 1e-6,
    tracerBudget: 0,
    transportExperiment,
    production: { dtS: dt, timeStep: "paper" },
  })) as WasmWorld;
  let graph: AdvanceGraph | undefined;
  const rows: Array<Record<string, unknown>> = [];
  let initialContourFine: number[] = [];
  let lastContourFine: number[] = [];
  let failure: { frame: number; error: string } | null = null;
  try {
    for (let frame = 0; frame <= frames; frame++) {
      try {
        const receipt = frame === 0
          ? parsePhysicsReceipt(world.receipt())
          : parsePhysicsReceipt(world.advance(frame, dt));
        const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
          bytes: world.snapshot(0xf).slice(), release() {} });
        let view: AdvanceView;
        try { view = createAdvanceView(decoded, graph, scene); }
        finally { decoded.release(); }
        graph = view.graph;
        const contour = Array.from(view.rdf.segmentsFine);
        if (frame === 0) initialContourFine = contour;
        lastContourFine = contour;
        const moving = arm !== "lsv-stationary";
        const expectedM: [number, number] = [-0.25 + (moving ? speed * dt * frame : 0), 0.6];
        const expectedFine: [number, number] = [view.nx / 2 + expectedM[0] / h, expectedM[1] / h];
        const nativeFrame = native.frames[frame];
        assert.ok(nativeFrame, `native arm stopped before SIMD frame ${frame}`);
        const nativeReceipt = nativeFrame.receipt;
        assert.ok(Math.abs(Number(nativeReceipt.liquidMeasure) - Number(receipt.liquidMeasure)) <= 1e-5,
          `frame ${frame}: native/SIMD liquid measure mismatch`);
        assert.equal(nativeReceipt.topologyGeneration, receipt.topologyGeneration,
          `frame ${frame}: native/SIMD topology generation mismatch`);
        const centroid = nativeFrame.liquidCentroid as [number, number] | null;
        const centroidM = centroid ? [(centroid[0] - view.nx / 2) * h, centroid[1] * h] : null;
        rows.push({
          frame,
          timeS: frame * dt,
          expectedCentroidM: expectedM,
          nativeLiquidMeasureFine: nativeReceipt.liquidMeasure,
          nativeRelativeMassDrift: nativeReceipt.drift,
          nativeCentroidM: centroidM,
          nativeCentroidErrorM: centroidM
            ? Math.hypot(centroidM[0] - expectedM[0], centroidM[1] - expectedM[1]) : null,
          nativeLiquidMeanVelocityFinePerS: nativeFrame.liquidMeanVelocity,
          nativeLiquidVelocityCovarianceFinePerS: nativeFrame.liquidVelocityCovariance,
          nativeLiquidCovarianceFine: nativeFrame.liquidCovariance,
          densityAxisRatio: covarianceAxisRatio(nativeFrame.liquidCovariance),
          topologyGeneration: nativeReceipt.topologyGeneration,
          cellCount: nativeFrame.cellCount,
          cellWidths: nativeFrame.cellWidths,
          cellsByWidth: nativeFrame.cellsByWidth,
          mixedSeamCount: nativeFrame.mixedSeamCount,
          wetMixedSeamCount: nativeFrame.wetMixedSeamCount,
          levelSetVolume: nativeReceipt.levelSetVolume ?? null,
          interfaceSeams: nativeReceipt.interfaceSeams,
          rdfReceipt: view.rdf.receipt,
          surface: surfaceMetrics(view, expectedFine),
        });
      } catch (error) {
        failure = { frame, error: error instanceof Error ? error.stack ?? error.message : String(error) };
        break;
      }
    }
  } finally { world.free(); }
  return { rows, failure, contours: { initialFine: initialContourFine, finalFine: lastContourFine } };
}

function correlation(rows: Array<Record<string, unknown>>, x: (row: Record<string, unknown>) => number,
  y: (row: Record<string, unknown>) => number): number | null {
  const pairs = rows.map(row => [x(row), y(row)] as const).filter(pair => pair.every(Number.isFinite));
  if (pairs.length < 2) return null;
  const mx = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const my = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  let xy = 0, xx = 0, yy = 0;
  for (const [a, b] of pairs) { xy += (a - mx) * (b - my); xx += (a - mx) ** 2; yy += (b - my) ** 2; }
  return xx > 0 && yy > 0 ? xy / Math.sqrt(xx * yy) : null;
}

const results = [];
for (const arm of arms) {
  const { document } = configuredScene(arm);
  const native = nativeRun(arm, document);
  const wasm = await wasmRun(arm, document, native);
  const rows = wasm.rows;
  const final = rows.at(-1);
  const topologyChanges = rows.slice(1).filter((row, index) =>
    row.topologyGeneration !== rows[index]!.topologyGeneration).length;
  results.push({
    arm,
    requestedFrames: frames,
    nativeFailure: native.failure,
    wasmFailure: wasm.failure,
    producedFrames: rows.length - 1,
    initial: rows[0],
    final,
    extrema: {
      maximumCentroidErrorM: Math.max(...rows.map(row => Number(row.nativeCentroidErrorM))),
      minimumCircularity: Math.min(...rows.map(row => Number((row.surface as Record<string, unknown>).circularity))),
      maximumRadialRmsErrorM: Math.max(...rows.map(row => Number((row.surface as Record<string, unknown>).radialRmsErrorM))),
      maximumBoundaryAxisRatio: Math.max(...rows.map(row => Number((row.surface as Record<string, unknown>).boundaryAxisRatio))),
      maximumDensityAxisRatio: Math.max(...rows.map(row => Number(row.densityAxisRatio))),
    },
    topologyChanges,
    correlations: {
      topologyGenerationVsRadialError: correlation(rows, row => Number(row.topologyGeneration),
        row => Number((row.surface as Record<string, unknown>).radialRmsErrorM)),
      overcapacityVsRadialError: correlation(rows,
        row => Number((row.levelSetVolume as Record<string, unknown> | null)?.totalVolumeOverCapacity ?? 0),
        row => Number((row.surface as Record<string, unknown>).radialRmsErrorM)),
      rowResidualVsRadialError: correlation(rows,
        row => Number((row.levelSetVolume as Record<string, unknown> | null)?.maximumNormalizedRowResidual ?? 0),
        row => Number((row.surface as Record<string, unknown>).radialRmsErrorM)),
    },
    contours: wasm.contours,
    rows,
  });
}

const report = {
  scene: "gentle-moving-blob",
  engines: { dynamics: "native-release-verify_world", surface: "simd-wasm-ui-publication" },
  configuration: { frames, dtS: dt, durationS: frames * dt, pressureIterations: 256,
    finestCellSizeM: h, speedMPerS: speed },
  analytic: {
    authoredSphereRadiusM: authoredRadius,
    selectedSliceZM: sliceZ,
    selectedSliceRadiusM: sectionRadius,
    selectedSliceAreaM2: Math.PI * sectionRadius * sectionRadius,
    expectedTranslationM: speed * frames * dt,
    expectedFinestCellsPerFrame: speed * dt / h,
    predictedLinearGatherVarianceIncreaseM2: frames * (speed * dt / h)
      * (1 - speed * dt / h) * h * h,
  },
  arms: results,
};
const serialized = `${JSON.stringify(report, null, 2)}\n`;
writeFileSync(resolve(root, outputPath), serialized);
process.stdout.write(`${JSON.stringify({
  output: outputPath,
  arms: results.map(result => ({ arm: result.arm, producedFrames: result.producedFrames,
    nativeFailure: result.nativeFailure, wasmFailure: result.wasmFailure, final: result.final,
    extrema: result.extrema, topologyChanges: result.topologyChanges, correlations: result.correlations })),
}, null, 2)}\n`);
