/** Diagnose shared-level-set flatness for the cell-cut hydrostatic oracle. */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { createAdvanceView, type AdvanceGraph, type AdvanceView } from "../lib/physics-wasm/advance-view";
import { decodePhysicsPublication } from "../lib/physics-wasm/publication";
import { parsePhysicsReceipt } from "../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./wasm/load-module.mjs";

type WasmWorld = {
  advance(sequence: number, dt: number): string;
  receipt(): string;
  snapshot(mask: number): Uint8Array;
  free(): void;
};

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const outputPath = resolve(root, process.argv.find(value => value.startsWith("--output="))
  ?.slice("--output=".length) ?? "artifacts/level-set-volume/hydrostatic-power-large-offset-lsv-surface.json");
const frames = Number(process.argv.find(value => value.startsWith("--frames="))
  ?.slice("--frames=".length) ?? "30");
assert.ok(Number.isInteger(frames) && frames >= 1);

const dt = 1 / 30;
const expectedSurfaceY = 15.25;

const summary = (values: readonly number[]) => {
  if (values.length === 0) return null;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    count: values.length,
    minimum: Math.min(...values), maximum: Math.max(...values), mean,
    rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length),
  };
};

function diagnose(view: AdvanceView, initialPhi?: Float32Array, previousPhi?: Float32Array) {
  const phi = view.rdf.vertexPhiFine;
  const stride = view.nx + 1;
  const crossings: Array<{ x: number; y: number; error: number }> = [];
  for (let x = 0; x <= view.nx; x++) {
    for (let y = 0; y < view.ny; y++) {
      const a = phi[x + stride * y]!, b = phi[x + stride * (y + 1)]!;
      if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0)) {
        if (a === b) continue;
        const crossing = y + a / (a - b);
        crossings.push({ x, y: crossing, error: crossing - expectedSurfaceY });
      }
    }
  }

  const segmentRows: Record<string, number[]> = {};
  const segmentErrors: number[] = [];
  const interiorSegmentErrors: number[] = [];
  const segmentSlopes: number[] = [];
  const segments = view.rdf.segmentsFine;
  for (let at = 0; at + 3 < segments.length; at += 4) {
    const x0 = segments[at]!, y0 = segments[at + 1]!;
    const x1 = segments[at + 2]!, y1 = segments[at + 3]!;
    const mx = 0.5 * (x0 + x1), my = 0.5 * (y0 + y1);
    const cell = view.lattice.cells.find(value => mx >= value.x0 && mx <= value.x0 + value.width
      && my >= value.y0 && my <= value.y0 + value.height);
    const width = String(cell?.width ?? -1);
    (segmentRows[width] ??= []).push(my - expectedSurfaceY);
    segmentErrors.push(my - expectedSurfaceY);
    if (mx >= 1 && mx <= view.nx - 1) interiorSegmentErrors.push(my - expectedSurfaceY);
    if (Math.abs(x1 - x0) > 1e-12) segmentSlopes.push((y1 - y0) / (x1 - x0));
  }

  let totalVolume = 0, expectedVolume = 0, volumeL1 = 0, maximumFillError = 0;
  const volumeErrorByWidth: Record<string, number> = {};
  for (const cell of view.lattice.cells) {
    const graphCell = view.graph.cells[cell.topologyCell]!;
    const represented = cell.volume;
    const physicalY0 = graphCell.minimum[1]!;
    const physicalY1 = graphCell.maximum[1]!;
    const overlapHeight = Math.max(0, Math.min(physicalY1, expectedSurfaceY) - physicalY0);
    const expected = overlapHeight * graphCell.widths[0]!;
    const error = represented - expected;
    totalVolume += represented;
    expectedVolume += expected;
    volumeL1 += Math.abs(error);
    maximumFillError = Math.max(maximumFillError, Math.abs(cell.fill - expected / graphCell.measure));
    volumeErrorByWidth[cell.width] = (volumeErrorByWidth[cell.width] ?? 0) + Math.abs(error);
  }
  let maximumPhiChange = 0, rmsPhiChange = 0;
  const largestPhiChanges: Array<{ x: number; y: number; initial: number; current: number; delta: number }> = [];
  if (initialPhi) {
    let squared = 0;
    for (let i = 0; i < phi.length; i++) {
      const delta = phi[i]! - initialPhi[i]!;
      maximumPhiChange = Math.max(maximumPhiChange, Math.abs(delta));
      squared += delta * delta;
      if (Math.abs(delta) > 1e-7) largestPhiChanges.push({ x: i % stride, y: Math.floor(i / stride),
        initial: initialPhi[i]!, current: phi[i]!, delta });
    }
    rmsPhiChange = Math.sqrt(squared / phi.length);
  }
  const changedFromPrevious: Array<{ x: number; y: number; previous: number; current: number; delta: number }> = [];
  if (previousPhi) for (let i = 0; i < phi.length; i++) {
    const delta = phi[i]! - previousPhi[i]!;
    if (Math.abs(delta) > 1e-7) changedFromPrevious.push({ x: i % stride, y: Math.floor(i / stride),
      previous: previousPhi[i]!, current: phi[i]!, delta });
  }
  return {
    contour: {
      segmentCount: segments.length / 4,
      midpointErrorFine: summary(segmentErrors),
      interiorMidpointErrorFine: summary(interiorSegmentErrors),
      slope: summary(segmentSlopes),
      verticalCrossingErrorFine: summary(crossings.map(value => value.error)),
      errorByUnderlyingCellWidth: Object.fromEntries(Object.entries(segmentRows)
        .map(([width, errors]) => [width, summary(errors)])),
    },
    volume: { totalFine: totalVolume, expectedFine: expectedVolume,
      signedErrorFine: totalVolume - expectedVolume, l1CellErrorFine: volumeL1,
      maximumFillError, l1ErrorByCellWidth: volumeErrorByWidth },
    phiChangeFromInitial: initialPhi ? { maximum: maximumPhiChange, rms: rmsPhiChange } : null,
    largestPhiChanges: largestPhiChanges.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 16),
    changeFromPrevious: {
      count: changedFromPrevious.length,
      largest: changedFromPrevious.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 32),
      nearAuthoredSurface: changedFromPrevious.filter(value => value.y >= 13 && value.y <= 19),
    },
    phiRowsNearAuthoredSurface: Object.fromEntries([14, 15, 16, 17, 18]
      .map(y => [y, summary(Array.from({ length: view.nx + 1 }, (_, x) => phi[x + stride * y]!))])),
    contourSegmentsFine: Array.from(segments),
  };
}

const definition = findSceneDefinition("hydrostatic-power-large-offset");
assert.ok(definition);
const document = sceneDocument(definition);
const scene = { id: definition.id, label: definition.name, document };
const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
const world = wasm.FluidWorld.from_scene(JSON.stringify(document), JSON.stringify({
  runEpoch: 1, commandSequence: 0, pressureIterations: 256,
  pressureRelativeTolerance: 1e-6, tracerBudget: 0,
  transportExperiment: "level-set-volume",
  production: { dtS: dt, timeStep: "paper" },
})) as WasmWorld;

let graph: AdvanceGraph | undefined;
let initialPhi: Float32Array | undefined;
let previousPhi: Float32Array | undefined;
const rows: unknown[] = [];
try {
  for (let frame = 0; frame <= frames; frame++) {
    const receipt = parsePhysicsReceipt(frame === 0 ? world.receipt() : world.advance(frame, dt));
    const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
      bytes: world.snapshot(0xf).slice(), release() {} });
    try {
      const view = createAdvanceView(decoded, graph, scene);
      graph = view.graph;
      if (!initialPhi) initialPhi = view.rdf.vertexPhiFine.slice();
      const widths = Object.fromEntries([1, 2, 4, 8].map(width => [width,
        view.graph.cells.filter(cell => cell.widths[0] === width).length]));
      const velocities = [...view.faceVelocityXFine, ...view.faceVelocityYFine];
      rows.push({ frame, timeS: frame * dt, topologyGeneration: receipt.topologyGeneration,
        liquidMeasure: receipt.liquidMeasure,
        publishedFaceVelocityFinePerS: {
          maximumAbsolute: velocities.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0),
          rms: Math.sqrt(velocities.reduce((sum, value) => sum + value * value, 0) / velocities.length),
        },
        pressure: receipt.pressure, levelSetVolume: receipt.levelSetVolume,
        cellCount: view.graph.cells.length, cellsByWidth: widths,
        ...diagnose(view, initialPhi, previousPhi) });
      previousPhi = view.rdf.vertexPhiFine.slice();
    } finally { decoded.release(); }
  }
} finally { world.free(); }

const build = JSON.parse(readFileSync(resolve(root,
  "public/wasm/fluid-wasm/simd/build-info.json"), "utf8"));
mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, JSON.stringify({ sceneId: definition.id, dt, frames,
  expectedSurfaceYFine: expectedSurfaceY, expectedSurfaceYM: expectedSurfaceY * 0.05,
  wasmBuild: build, rows }, null, 2) + "\n");
console.log(outputPath);
