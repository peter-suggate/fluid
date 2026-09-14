import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { createAdvanceView, type AdvanceGraph } from "../lib/physics-wasm/advance-view";
import { decodePhysicsPublication, PhysicsPlane } from "../lib/physics-wasm/publication";
import { parsePhysicsReceipt } from "../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./wasm/load-module.mjs";

const outputPath = resolve(process.argv.find(value => value.startsWith("--output="))?.slice(9)
  ?? "artifacts/level-set-volume/cm12-figure-7-lsv-resolution-frame0-6.json");
const activityBits = new Map<number, string>([
  [1 << 0, "surface"], [1 << 1, "deformation"], [1 << 2, "temporal"],
  [1 << 3, "fine-detail"], [1 << 4, "predicted-face"], [1 << 5, "first-step"],
  [1 << 6, "occupied"], [1 << 7, "velocity-floor"], [1 << 8, "thin-fluid"],
  [1 << 9, "cut-boundary"], [1 << 10, "static-boundary-8"],
  [1 << 11, "static-boundary-4"], [1 << 12, "static-boundary-2"],
  [1 << 13, "static-boundary-1"], [1 << 14, "density-surface"],
]);
const planReasons = new Map<number, string>([
  [2, "page-or-predicted-face-demand"], [4, "required-resolution-floor"],
  [8, "hot-promotion"], [16, "quiet-or-surface-proof-demotion"],
  [32, "hold"], [64, "velocity-floor-promotion"], [128, "inactive"],
  [256, "thin-fluid-promotion"], [2048, "deeply-enclosed-demotion"],
  [0x8000_0000, "allocated-inactive"], [0x8000_0001, "allocated-active"],
]);

const count = (values: readonly number[]): Record<string, number> => values.reduce<Record<string, number>>(
  (result, value) => { const key = String(value); result[key] = (result[key] ?? 0) + 1; return result; }, {});
const decodeBits = (bits: number): string[] => [...activityBits]
  .filter(([mask]) => (bits & mask) !== 0).map(([, label]) => label);
const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
function brickMotion(graph: AdvanceGraph, key: number, density: Float32Array,
  velocity: Float32Array, dt: number): Record<string, unknown> {
  const ids = graph.cells.filter(cell => cell.brickKey === key && density[cell.id]! > 0).map(cell => cell.id);
  let weight = 0, mx = 0, my = 0;
  for (const id of ids) {
    const w = density[id]!; weight += w; mx += w * velocity[2 * id]!; my += w * velocity[2 * id + 1]!;
  }
  const mean = weight > 0 ? [mx / weight, my / weight] : [0, 0];
  let maximumAbsoluteTravel = 0, maximumRelativeTravel = 0;
  const x: number[] = [], y: number[] = [];
  for (const id of ids) {
    const vx = velocity[2 * id]!, vy = velocity[2 * id + 1]!;
    x.push(vx); y.push(vy);
    maximumAbsoluteTravel = Math.max(maximumAbsoluteTravel, dt * Math.hypot(vx, vy));
    maximumRelativeTravel = Math.max(maximumRelativeTravel,
      dt * Math.hypot(vx - mean[0]!, vy - mean[1]!));
  }
  return { wetCellCount: ids.length, densityWeightedMeanVelocityFinePerS: mean,
    velocityRangeFinePerS: ids.length ? { x: [Math.min(...x), Math.max(...x)],
      y: [Math.min(...y), Math.max(...y)] } : null,
    reconstructedMaximumAbsoluteTravelFine: maximumAbsoluteTravel,
    reconstructedMaximumRelativeTravelFine: maximumRelativeTravel };
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

const frames: Record<string, unknown>[] = [];
let graph: AdvanceGraph | undefined;
let priorBricks = new Map<number, { resolution: number; active: boolean }>();
try {
  for (let frame = 0; frame <= 6; frame++) {
    const receipt = frame === 0 ? parsePhysicsReceipt(world.receipt())
      : parsePhysicsReceipt(world.advance(frame, 1 / 30));
    const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
      bytes: world.snapshot(0xf).slice(), release() {} });
    try {
      const view = createAdvanceView(decoded, graph, scene);
      graph = view.graph;
      const currentBrickResolution = new Map(view.graph.bricks.map(brick => [brick.key, brick.resolution]));
      const density = decoded.plane(PhysicsPlane.Density) as Float32Array;
      const velocity = decoded.plane(PhysicsPlane.CellVelocity) as Float32Array;
      const resolution = asRecord(decoded.metadata.resolution);
      const brickReceipts = Array.isArray(resolution?.bricks) ? resolution.bricks
        .map(asRecord).filter((value): value is Record<string, unknown> => value !== null) : [];
      const receiptByKey = new Map(brickReceipts.map(brick => [Number(brick.brickKey), brick]));
      const transitions = brickReceipts.flatMap(brick => {
        const key = Number(brick.brickKey), accepted = Number(brick.acceptedResolution);
        const requested = Number(brick.requestedResolution), scheduled = Number(brick.scheduledResolution);
        const current = currentBrickResolution.get(key), prior = priorBricks.get(key);
        const candidateActive = Boolean(brick.candidateActive);
        if (accepted === requested && requested === scheduled && current === prior?.resolution
          && candidateActive === prior?.active) return [];
        const reasons = Number(brick.reasons), plan = Number(brick.planReasons);
        return [{ brickKey: key, priorResolution: prior?.resolution ?? null, graphResolution: current ?? null,
          acceptedResolution: accepted, requestedResolution: requested, scheduledResolution: scheduled,
          acceptedActive: Boolean(brick.acceptedActive), candidateActive,
          scoreByte: Number(brick.scoreByte), reasons, reasonLabels: decodeBits(reasons),
          planReasons: plan, planReasonLabel: planReasons.get(plan) ?? `unknown-${plan}`,
          supportMask: Number(brick.supportMask), sweptSupportMask: Number(brick.sweptSupportMask),
          faultBits: Number(brick.faultBits) }];
      });
      const bulkWidth4Bricks = view.graph.bricks.filter(brick => brick.active && brick.resolution === 2)
        .map(brick => {
          const record = receiptByKey.get(brick.key);
          const reasons = Number(record?.reasons ?? 0), plan = Number(record?.planReasons ?? 0);
          return { brickKey: brick.key, scoreByte: record ? Number(record.scoreByte) : null,
            reasons, reasonLabels: decodeBits(reasons), planReasons: record ? plan : null,
            planReasonLabel: record ? planReasons.get(plan) ?? `unknown-${plan}` : null,
            ...brickMotion(view.graph, brick.key, density, velocity, 1 / 30) };
        });
      frames.push({ frame, timeS: receipt.time, topologyGeneration: receipt.topologyGeneration,
        liquidMeasure: receipt.liquidMeasure, fault: receipt.fault,
        activeBricksByResolution: count(view.graph.bricks.filter(brick => brick.active)
          .map(brick => brick.resolution)),
        cellsByWidth: count(view.graph.cells.map(cell => cell.widths[0]!)),
        resolutionSummary: resolution && {
          topologyEpoch: resolution.topologyEpoch, acceptedGeneration: resolution.acceptedGeneration,
          candidateGeneration: resolution.candidateGeneration, measuredBrickCount: resolution.measuredBrickCount,
          surfaceBrickCount: resolution.surfaceBrickCount, occupiedBrickCount: resolution.occupiedBrickCount,
          promotedBrickCount: resolution.promotedBrickCount, demotedBrickCount: resolution.demotedBrickCount,
          deferredDemotionCount: resolution.deferredDemotionCount, maximumScoreByte: resolution.maximumScoreByte,
          faultBits: resolution.faultBits,
        }, bulkWidth4Bricks, transitions });
      priorBricks = new Map(view.graph.bricks.map(brick => [brick.key,
        { resolution: brick.resolution, active: brick.active }]));
    } finally { decoded.release(); }
  }
} finally { world.free(); }

const report = { scene: "cm12-figure-7", transportExperiment: "level-set-volume",
  artifact: "simd", pressureIterations: 256, pressureRelativeTolerance: 1e-6,
  dtS: 1 / 30, frames };
mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ outputPath, frames: frames.map(({ frame, timeS, topologyGeneration,
  activeBricksByResolution, cellsByWidth, bulkWidth4Bricks, transitions }) => ({ frame, timeS,
  topologyGeneration, activeBricksByResolution, cellsByWidth, bulkWidth4Bricks, transitions })) }, null, 2));
