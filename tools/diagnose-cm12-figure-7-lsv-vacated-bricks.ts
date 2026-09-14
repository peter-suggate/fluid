import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { createAdvanceView, type AdvanceGraph } from "../lib/physics-wasm/advance-view";
import { decodePhysicsPublication, PhysicsPlane } from "../lib/physics-wasm/publication";
import { parsePhysicsReceipt } from "../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./wasm/load-module.mjs";

type Datum = Record<string, unknown> & {
  frame: number; brickKey: number; resolution: number; active: boolean;
  totalVolumeFine: number; totalCapacityFine: number; volumeCapacityRatio: number;
  contourSegmentCount: number; minimumVertexPhiFine: number; maximumFill: number;
};

const root = resolve(import.meta.dirname, "..");
const outputPath = resolve(process.argv.find(value => value.startsWith("--output="))?.slice(9)
  ?? "artifacts/level-set-volume/cm12-figure-7-lsv-vacated-fine-bricks-frame0-30.json");
const wasmPath = resolve(root, "public/wasm/fluid-wasm/simd/fluid_wasm_bg.wasm");
const sha256 = createHash("sha256").update(readFileSync(wasmPath)).digest("hex");
const record = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;

function segmentIntersectsBox(ax: number, ay: number, bx: number, by: number,
  loX: number, loY: number, hiX: number, hiY: number): boolean {
  let lower = 0, upper = 1;
  for (const [p, q] of [[-(bx - ax), ax - loX], [bx - ax, hiX - ax],
    [-(by - ay), ay - loY], [by - ay, hiY - ay]] as const) {
    if (p === 0) { if (q < 0) return false; continue; }
    const ratio = q / p;
    if (p < 0) lower = Math.max(lower, ratio); else upper = Math.min(upper, ratio);
    if (lower > upper) return false;
  }
  return true;
}

function sampleBrick(frame: number, graph: AdvanceGraph, brick: AdvanceGraph["bricks"][number],
  density: Float32Array, capacity: Float32Array, velocity: Float32Array, extension: Uint8Array,
  phi: Float32Array, segments: Float32Array,
  receiptByKey: ReadonlyMap<number, Record<string, unknown>>): Datum {
  const loX = brick.coordinate[0]! * 8, loY = brick.coordinate[1]! * 8;
  const hiX = loX + brick.spanBricks * 8, hiY = loY + brick.spanBricks * 8;
  const cells = graph.cells.filter(cell => cell.brickKey === brick.key);
  let totalVolumeFine = 0, totalCapacityFine = 0, maximumFill = 0;
  let velocityWeight = 0, meanVelocityX = 0, meanVelocityY = 0, maximumSpeed = 0;
  const extensionDepthCounts: Record<string, number> = {};
  for (const cell of cells) {
    const volume = density[cell.id]! * cell.measure, available = capacity[cell.id]! * cell.measure;
    totalVolumeFine += volume; totalCapacityFine += available;
    if (available > 1e-8) maximumFill = Math.max(maximumFill, volume / available);
    if (volume > 0) {
      const vx = velocity[2 * cell.id]!, vy = velocity[2 * cell.id + 1]!;
      velocityWeight += volume; meanVelocityX += volume * vx; meanVelocityY += volume * vy;
      maximumSpeed = Math.max(maximumSpeed, Math.hypot(vx, vy));
    }
    const depth = String(extension[cell.id]!);
    extensionDepthCounts[depth] = (extensionDepthCounts[depth] ?? 0) + 1;
  }
  const stride = graph.dimensions[0]! + 1;
  let minimumVertexPhiFine = Infinity, maximumVertexPhiFine = -Infinity;
  for (let y = loY; y <= hiY; y++) for (let x = loX; x <= hiX; x++) {
    const value = phi[x + stride * y]!;
    minimumVertexPhiFine = Math.min(minimumVertexPhiFine, value);
    maximumVertexPhiFine = Math.max(maximumVertexPhiFine, value);
  }
  let contourSegmentCount = 0;
  for (let at = 0; at + 3 < segments.length; at += 4) {
    if (segmentIntersectsBox(segments[at]!, segments[at + 1]!, segments[at + 2]!, segments[at + 3]!,
      loX, loY, hiX, hiY)) contourSegmentCount++;
  }
  const receipt = receiptByKey.get(brick.key);
  const reasons = Number(receipt?.reasons ?? 0), planReasons = Number(receipt?.planReasons ?? 0);
  return { frame, brickKey: brick.key, coordinate: brick.coordinate, resolution: brick.resolution,
    active: brick.active, cellCount: cells.length, totalVolumeFine, totalCapacityFine,
    volumeCapacityRatio: totalCapacityFine > 0 ? totalVolumeFine / totalCapacityFine : 0,
    maximumFill, minimumVertexPhiFine, maximumVertexPhiFine, contourSegmentCount,
    volumeWeightedMeanVelocityFinePerS: velocityWeight > 0
      ? [meanVelocityX / velocityWeight, meanVelocityY / velocityWeight] : null,
    maximumMaterialCellSpeedFinePerS: maximumSpeed, extensionDepthCounts,
    phiAllPositive: minimumVertexPhiFine > 0, surfaceReason: (reasons & 1) !== 0,
    thinFluidReason: (reasons & (1 << 8)) !== 0,
    predictedFaceReason: (reasons & (1 << 4)) !== 0,
    velocityFloorReason: (reasons & (1 << 7)) !== 0,
    curvatureFloor: reasons >>> 16, scoreByte: receipt ? Number(receipt.scoreByte) : null,
    acceptedResolution: receipt ? Number(receipt.acceptedResolution) : null,
    requestedResolution: receipt ? Number(receipt.requestedResolution) : null,
    scheduledResolution: receipt ? Number(receipt.scheduledResolution) : null,
    planReasons: receipt ? planReasons : null,
    pageOrPredictedFaceDemand: planReasons === 2,
    quietOrSurfaceProofDemotion: planReasons === 16,
    surfaceProofEpochs: null,
    surfaceProofNote: "proof epochs are internal policy state and are not published in the Wasm resolution receipt",
  };
}

const wasm = await loadFluidWasmForNode(undefined, { artifact: "simd" });
const definition = findSceneDefinition("cm12-figure-7");
assert.ok(definition);
const scene = { id: definition.id, label: definition.name, document: sceneDocument(definition) };
const world = wasm.FluidWorld.from_scene(JSON.stringify(scene.document), JSON.stringify({
  runEpoch: 1, commandSequence: 0, pressureIterations: 256,
  pressureRelativeTolerance: 1e-6, transportExperiment: "level-set-volume",
  production: { dtS: 1 / 30, timeStep: "paper" },
}));
const histories = new Map<number, Datum[]>(), frameSummaries: Record<string, unknown>[] = [];
const frameStates: Record<string, unknown>[] = [];
let graph: AdvanceGraph | undefined;
try {
  for (let frame = 0; frame <= 30; frame++) {
    const receipt = frame === 0 ? parsePhysicsReceipt(world.receipt())
      : parsePhysicsReceipt(world.advance(frame, 1 / 30));
    const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
      bytes: world.snapshot(0xf).slice(), release() {} });
    try {
      const view = createAdvanceView(decoded, graph, scene); graph = view.graph;
      const density = decoded.plane(PhysicsPlane.Density) as Float32Array;
      const capacity = decoded.plane(PhysicsPlane.Capacity) as Float32Array;
      const velocity = decoded.plane(PhysicsPlane.CellVelocity) as Float32Array;
      const extension = decoded.plane(PhysicsPlane.ExtensionDepth) as Uint8Array;
      const resolution = record(decoded.metadata.resolution);
      const receipts = Array.isArray(resolution?.bricks) ? resolution.bricks.map(record)
        .filter((value): value is Record<string, unknown> => value !== null) : [];
      const receiptByKey = new Map(receipts.map(value => [Number(value.brickKey), value]));
      const data = view.graph.bricks.map(brick => sampleBrick(frame, view.graph, brick, density, capacity,
        velocity, extension, view.rdf.vertexPhiFine, view.rdf.segmentsFine, receiptByKey));
      for (const datum of data) histories.set(datum.brickKey, [...histories.get(datum.brickKey) ?? [], datum]);
      frameSummaries.push({ frame, timeS: receipt.time, topologyGeneration: receipt.topologyGeneration,
        activeCells: view.graph.cells.length,
        liquidMeasure: receipt.liquidMeasure,
        levelSetVolume: receipt.levelSetVolume,
        pressure: receipt.pressure,
        surface: view.rdf.receipt,
        excessVolumeFine: view.graph.cells.reduce((sum, cell) => sum
          + Math.max(0, density[cell.id]! - capacity[cell.id]!) * cell.measure, 0),
        contourLengthFine: Array.from({ length: view.rdf.segmentsFine.length / 4 }, (_, i) => {
          const at = 4 * i, segments = view.rdf.segmentsFine;
          return Math.hypot(segments[at + 2]! - segments[at]!, segments[at + 3]! - segments[at + 1]!);
        }).reduce((sum, length) => sum + length, 0),
        activeBricks: data.filter(value => value.active).length,
        activeFineBricks: data.filter(value => value.active && value.resolution === 8).length,
        activeFineNoContour: data.filter(value => value.active && value.resolution === 8
          && value.contourSegmentCount === 0).length,
        activeFinePhiAllPositive: data.filter(value => value.active && value.resolution === 8
          && value.minimumVertexPhiFine > 0).length,
        inactiveFineBricksWithCells: data.filter(value => !value.active && value.resolution === 8
          && Number(value.cellCount) > 0).length });
      frameStates.push({ frame, timeS: receipt.time, topologyGeneration: receipt.topologyGeneration,
        contourSegmentsFine: Array.from(view.rdf.segmentsFine), bricks: data });
    } finally { decoded.release(); }
  }
} finally { world.free(); }

const vacated = [...histories].flatMap(([brickKey, values]) => {
  const first = values.find((value, index) => values.slice(0, index)
    .some(prior => prior.totalVolumeFine > 1 || prior.contourSegmentCount > 0)
    && value.active && value.resolution === 8 && value.contourSegmentCount === 0
    && value.minimumVertexPhiFine > 0 && value.volumeCapacityRatio < 0.05);
  return first ? [{ brickKey, firstVacatedFrame: first.frame, history: values }] : [];
}).sort((a, b) => a.firstVacatedFrame - b.firstVacatedFrame || a.brickKey - b.brickKey);
const persistentAirFine = [...histories].flatMap(([brickKey, values]) => {
  let longest: Datum[] = [], run: Datum[] = [];
  for (const value of values) {
    const airFine = value.resolution === 8 && value.contourSegmentCount === 0
      && value.minimumVertexPhiFine > 0 && value.volumeCapacityRatio < 0.05;
    run = airFine ? [...run, value] : [];
    if (run.length > longest.length) longest = run;
  }
  return longest.length >= 3 ? [{ brickKey, longestRun: longest }] : [];
}).sort((a, b) => b.longestRun.length - a.longestRun.length || a.brickKey - b.brickKey);
const lateAirFine = [25, 27, 30].map(frame => ({ frame, bricks: [...histories]
  .flatMap(([brickKey, values]) => {
    const index = values.findIndex(value => value.frame === frame), value = values[index];
    if (!value?.active || value.resolution !== 8 || value.contourSegmentCount !== 0
      || value.minimumVertexPhiFine <= 0) return [];
    const prior = values[index - 1], next = values[index + 1];
    const direction = prior && next
      ? next.totalVolumeFine > value.totalVolumeFine && value.totalVolumeFine >= prior.totalVolumeFine
        ? "incoming"
        : prior.totalVolumeFine > value.totalVolumeFine && value.totalVolumeFine >= next.totalVolumeFine
          ? "trailing" : "indeterminate"
      : "boundary-of-capture";
    return [{ brickKey, direction, value, prior: prior ?? null, next: next ?? null }];
  }) }));
const adjacencyCases = [2229, 2234].map(brickKey => {
  const target = histories.get(brickKey)?.find(value => value.frame === 17);
  const coordinate = target?.coordinate as readonly number[] | undefined;
  const neighbors = coordinate ? [...histories].flatMap(([key, values]) => {
    const value = values.find(candidate => candidate.frame === 17);
    const at = value?.coordinate as readonly number[] | undefined;
    return value && at && Math.max(Math.abs(at[0]! - coordinate[0]!),
      Math.abs(at[1]! - coordinate[1]!)) <= 2 ? [{ brickKey: key, ...value }] : [];
  }) : [];
  return { frame: 17, brickKey, target: target ?? null, neighborhoodRadiusBricks: 2, neighbors };
});
const report = { scene: "cm12-figure-7", transportExperiment: "level-set-volume", artifact: "simd",
  wasmSha256: sha256, pressureIterations: 256, pressureRelativeTolerance: 1e-6, dtS: 1 / 30,
  classification: "previously material-bearing; active resolution 8; no contour intersection; all brick vertices phi>0; V/capacity<5%",
  frameSummaries, vacatedBricks: vacated, persistentAirFineBricks: persistentAirFine,
  lateAirFineBricks: lateAirFine, adjacencyCases, frameStates };
mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ outputPath, wasmSha256: sha256, frameSummaries,
  vacatedBricks: vacated.map(value => ({ brickKey: value.brickKey,
    firstVacatedFrame: value.firstVacatedFrame,
    first: value.history.find(frame => frame.frame === value.firstVacatedFrame) })),
  persistentAirFineBricks: persistentAirFine.map(value => ({ brickKey: value.brickKey,
    frames: value.longestRun.map(frame => frame.frame), active: value.longestRun.map(frame => frame.active),
    cellCount: value.longestRun.map(frame => frame.cellCount) })) }, null, 2));
