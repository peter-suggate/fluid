import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type { SparseCM12TransportExperiment } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

type ActivityBrick = Awaited<ReturnType<
  WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]
>>["bricks"][number];

const brickKey = (coordinate: readonly number[]) => coordinate.join(",");

function resolutionHistogram(
  bricks: readonly ActivityBrick[],
  field: "acceptedResolution" | "candidateResolution",
) {
  return Object.fromEntries([1, 2, 4, 8].map((resolution) => [resolution,
    bricks.filter((brick) => brick[field] === resolution).length])) as
    Record<"1" | "2" | "4" | "8", number>;
}

function deeplySubmergedBricks(bricks: readonly ActivityBrick[]): readonly ActivityBrick[] {
  const byCoordinate = new Map(bricks.map((brick) => [brickKey(brick.coordinate), brick]));
  const directions = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]] as const;
  return bricks.filter((brick) => brick.active && (brick.reasons & 64) !== 0
    && directions.every((direction) => {
      const coordinate = brick.coordinate.map((value, axis) =>
        value + direction[axis]!) as [number, number, number];
      if (coordinate.some((value) => value < 0 || value >= 8)) return true;
      const neighbor = byCoordinate.get(brickKey(coordinate));
      return neighbor?.active === true && neighbor.meanDensity >= 0.5;
    }));
}

function assertTwoToOne(
  bricks: readonly ActivityBrick[],
  field: "acceptedResolution" | "candidateResolution",
): void {
  const byCoordinate = new Map(bricks.map((brick) => [brickKey(brick.coordinate), brick]));
  for (const brick of bricks) for (let axis = 0; axis < 3; axis += 1) {
    const coordinate = [...brick.coordinate] as [number, number, number];
    coordinate[axis] += 1;
    const neighbor = byCoordinate.get(brickKey(coordinate));
    if (!neighbor) continue;
    const low = Math.min(brick[field], neighbor[field]);
    const high = Math.max(brick[field], neighbor[field]);
    assert.ok(high <= 2 * low,
      `${field} violates 2:1 at ${brickKey(brick.coordinate)}/${brickKey(coordinate)}`);
  }
}

function assertSubmergedCandidatesAreCoarsest(bricks: readonly ActivityBrick[]): void {
  const byCoordinate = new Map(bricks.map((brick) => [brickKey(brick.coordinate), brick]));
  const directions = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]] as const;
  for (const brick of bricks.filter((candidate) => candidate.planReasons === 2048)) {
    // Publication updates acceptedResolution in the commit pass but retains
    // the candidate receipt that produced it. Once both are equal, judging
    // that completed request against the post-commit neighbours would ask it
    // to describe the *next* coarsening epoch. Pending requests still retain
    // the pre-publication accepted level and can be checked exactly here.
    if (brick.candidateResolution === brick.acceptedResolution) continue;
    let minimum = 1;
    const neighbors: Array<Record<string, unknown>> = [];
    for (const direction of directions) {
      const coordinate = brick.coordinate.map((value, axis) =>
        value + direction[axis]!) as [number, number, number];
      const neighbor = byCoordinate.get(brickKey(coordinate));
      if (!neighbor) continue;
      neighbors.push({ coordinate, accepted: neighbor.acceptedResolution,
        candidate: neighbor.candidateResolution, planned: neighbor.plannedResolution,
        planReasons: neighbor.planReasons, active: neighbor.active });
      minimum = Math.max(minimum,
        Math.max(neighbor.candidateResolution, neighbor.acceptedResolution) / 2);
    }
    assert.equal(brick.candidateResolution, minimum,
      `submerged brick ${brickKey(brick.coordinate)} retained avoidable resolution: ${
        JSON.stringify({ minimum, accepted: brick.acceptedResolution,
          candidate: brick.candidateResolution, planned: brick.plannedResolution,
          planReasons: brick.planReasons, reasons: brick.reasons,
          meanDensity: brick.meanDensity, neighbors })}`);
  }
}

function densityFrontX(density: Float32Array, threshold: number): number {
  let front = -1;
  for (let z = 0; z < 64; z += 1) for (let y = 0; y < 64; y += 1) {
    for (let x = 0; x < 64; x += 1) {
      if (density[x + 64 * (y + 64 * z)]! > threshold) front = Math.max(front, x);
    }
  }
  return front;
}

function densityReceipt(density: Float32Array) {
  let mass = 0, maximum = 0, momentX = 0;
  for (let z = 0; z < 64; z += 1) for (let y = 0; y < 64; y += 1) {
    for (let x = 0; x < 64; x += 1) {
      const rho = Math.max(0, density[x + 64 * (y + 64 * z)]!);
      mass += rho; maximum = Math.max(maximum, rho); momentX += rho * (x + 0.5) / 64;
    }
  }
  return {
    mass,
    maximum,
    centerOfMassX: momentX / Math.max(mass, Number.MIN_VALUE),
    front: {
      trace: densityFrontX(density, 1e-3),
      surface: densityFrontX(density, 0.05),
      liquid: densityFrontX(density, 0.5),
    },
  };
}

function relativeDensityL1(reference: Float32Array, candidate: Float32Array): number {
  let difference = 0, scale = 0;
  for (let index = 0; index < reference.length; index += 1) {
    difference += Math.abs(candidate[index]! - reference[index]!);
    scale += Math.abs(reference[index]!);
  }
  return difference / Math.max(scale, Number.MIN_VALUE);
}

dawnTest("Sparse CM12 expands the 64-cubed mini-dam into dormant receivers",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-mini64-front-dawn.test.ts");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      device.pushErrorScope("validation");
      const scene = createMinimalPowerDamBreak64Scene();
      const structureExperiment = process.env.FLUID_SPARSE_CM12_STRUCTURE_EXPERIMENT as
        Exclude<SparseCM12TransportExperiment, "baseline"> | undefined;
      const createSolver = (resolutionMode: "adaptive" | "all-fine") => {
        const args = [device!, scene, "balanced", undefined, {
          resolutionMode,
          brickFineResolution: 8,
          // Begin the experimental arm deliberately over-refined so this
          // short front regression exercises live submerged coarsening rather
          // than only the already-graded construction topology.
          surfaceFineRings: resolutionMode === "adaptive" ? 8 : 1,
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            // Keep the 8^3 oracle physically fixed over this five-step window.
            topologyCadenceSteps: resolutionMode === "adaptive" ? 1 : 64,
          },
          timeStep: "paper",
        }, () => {}] as const;
        return structureExperiment
          ? WebGPUAdaptiveMassSolver.createTransportExperimentForQA(
            structureExperiment, ...args)
          : WebGPUAdaptiveMassSolver.createAsync(...args);
      };
      const adaptive = await createSolver("adaptive");
      const allFine = await createSolver("all-fine");
      try {
        const initialFields = await Promise.all([
          adaptive.readDiagnosticFields(), allFine.readDiagnosticFields(),
        ]);
        const initial = {
          adaptive: densityReceipt(initialFields[0].density),
          allFine: densityReceipt(initialFields[1].density),
        };
        assert.equal(initial.adaptive.front.surface, 39,
          "the regression must start at the authored mini-dam face");
        assert.deepEqual(initial.adaptive, initial.allFine,
          "adaptive and all-fine controls must start from the same physical field");
        const initialActivity = await adaptive.readGPUActivityPolicy();
        const initialStats = await adaptive.readStats();

        const trajectory: Array<{
          step: number;
          adaptive: ReturnType<typeof densityReceipt>;
          allFine: ReturnType<typeof densityReceipt>;
          relativeL1: number;
          activity: {
            activeMaximumFineCellX: number;
            topology: {
              prepared: number;
              committed: number;
              deferred: number;
              shadowGeneration: number;
              acceptedCells: number;
              accepted: Record<"1" | "2" | "4" | "8", number>;
              candidate: Record<"1" | "2" | "4" | "8", number>;
              deeplySubmerged: number;
              aggressiveSubmerged: number;
            };
          };
        }> = [];
        for (let step = 1; step <= 5; step += 1) {
          const time_s = step * CM12_PAPER_DT_S;
          assert.equal(adaptive.advanceTo(time_s, []), true);
          assert.equal(allFine.advanceTo(time_s, []), true);
          await device.queue.onSubmittedWorkDone();
          const [adaptiveFields, allFineFields] = await Promise.all([
            adaptive.readDiagnosticFields(), allFine.readDiagnosticFields(),
          ]);
          const activity = await adaptive.readGPUActivityPolicy();
          const stats = await adaptive.readStats();
          assertTwoToOne(activity.bricks, "acceptedResolution");
          assertTwoToOne(activity.bricks, "candidateResolution");
          assertSubmergedCandidatesAreCoarsest(activity.bricks);
          const activeBricks = activity.bricks.filter((brick) => brick.active);
          const deeplySubmerged = deeplySubmergedBricks(activity.bricks);
          trajectory.push({
            step,
            adaptive: densityReceipt(adaptiveFields.density),
            allFine: densityReceipt(allFineFields.density),
            relativeL1: relativeDensityL1(allFineFields.density, adaptiveFields.density),
            activity: {
              activeMaximumFineCellX: Math.max(...activeBricks.map(
                (brick) => 8 * (brick.coordinate[0] + 1) - 1)),
              topology: {
                prepared: stats.adaptiveTopologyPreparedBrickCount ?? 0,
                committed: stats.adaptiveTopologyCommittedBrickCount ?? 0,
                deferred: stats.adaptiveTopologyDeferredBrickCount ?? 0,
                shadowGeneration: stats.adaptiveTopologyShadowGeneration ?? 0,
                acceptedCells: stats.adaptiveAcceptedCellCount ?? 0,
                accepted: resolutionHistogram(activity.bricks, "acceptedResolution"),
                candidate: resolutionHistogram(activity.bricks, "candidateResolution"),
                deeplySubmerged: deeplySubmerged.length,
                aggressiveSubmerged: activity.bricks.filter(
                  (brick) => brick.planReasons === 2048).length,
              },
            },
          });
        }
        const final = trajectory.at(-1)!;
        const maximumMassDrift = Math.max(...trajectory.flatMap((sample) => [
          Math.abs(sample.adaptive.mass - initial.adaptive.mass) / initial.adaptive.mass,
          Math.abs(sample.allFine.mass - initial.allFine.mass) / initial.allFine.mass,
        ]));
        assert.ok(maximumMassDrift <= 2e-3,
          `mini-dam mass drift must stay below 0.2%; measured ${maximumMassDrift}`);
        assert.ok(final.adaptive.front.surface >= 56,
          `the physical front must cross two dry brick columns; measured x=${final.adaptive.front.surface}`);
        assert.ok(trajectory.every((sample) =>
          Math.abs(sample.adaptive.front.surface - sample.allFine.front.surface) <= 1),
        "adaptive/all-fine surface fronts must agree within one fine cell at every frame");
        assert.ok(trajectory.every((sample) =>
          Math.abs(sample.adaptive.front.liquid - sample.allFine.front.liquid) <= 1),
        "adaptive/all-fine liquid fronts must agree within one fine cell at every frame");
        assert.ok(final.relativeL1 <= 0.06,
          `adaptive/all-fine density relative L1 ${final.relativeL1} exceeds 0.06`);
        assert.ok(Math.max(...trajectory.map((sample) => sample.adaptive.maximum)) <= 2,
          "adaptive density peak must stay bounded through the receiver transition");
        assert.ok(trajectory.every((sample, index) => index === 0
          || sample.adaptive.front.surface >= trajectory[index - 1]!.adaptive.front.surface),
        "adaptive surface front must not retreat during the five-frame release");
        assert.ok(trajectory.every((sample, index) => index === 0
          || sample.activity.topology.shadowGeneration
            > trajectory[index - 1]!.activity.topology.shadowGeneration),
        "a valid shadow topology must publish on every moving-front frame");
        assert.ok(trajectory.every((sample) => sample.activity.topology.prepared
          === sample.activity.topology.committed),
        "every prepared mini-dam transition must pass its conservation receipts");
        const finalActivity = await adaptive.readGPUActivityPolicy();
        const finalSubmerged = deeplySubmergedBricks(finalActivity.bricks);
        assert.ok(finalSubmerged.length > 0,
          "the mini-dam regression must retain a deeply submerged bulk region");
        assert.ok(trajectory.some((sample) =>
          sample.activity.topology.aggressiveSubmerged > 0),
        "the mini-dam release never exercised aggressive submerged coarsening");
        assert.ok((final.activity.topology.acceptedCells)
          < (initialStats.adaptiveAcceptedCellCount ?? Number.POSITIVE_INFINITY),
        "aggressive submerged coarsening must reduce accepted pressure-cell work");
        const finalStats = await adaptive.readStats();
        assert.ok((finalStats.adaptiveAcceptedSameLevelCoarseRowCount ?? 0) > 0,
          "accepted pressure topology must publish live same-level coarse rows");
        assert.ok((finalStats.adaptiveAcceptedMixedSeamRowCount ?? 0) > 0,
          "accepted pressure topology must publish live mixed-resolution rows");
        assert.ok((finalStats.adaptivePressureActiveRowCount ?? 0) > 0,
          "pressure classification must publish rows that entered the solve");
        assert.ok((finalStats.adaptivePressureActiveRowCount ?? Number.POSITIVE_INFINITY)
          <= (finalStats.adaptiveAcceptedRowCount ?? 0),
        "pressure-active rows must be a subset of the accepted topology");
        assert.equal(finalStats.adaptiveMixedSeamFaceCount,
          finalStats.adaptiveAcceptedMixedSeamRowCount,
        "the legacy mixed-seam receipt must track the live accepted census");
        if (process.env.FLUID_MINI64_FRONT_DIAGNOSTICS === "1") {
          console.log(JSON.stringify({
            phase: "sparse-cm12-mini64-front",
            initial: {
              receipt: initial.adaptive,
              acceptedCells: initialStats.adaptiveAcceptedCellCount,
              accepted: resolutionHistogram(initialActivity.bricks, "acceptedResolution"),
            },
            trajectory,
          }));
        }
        const activity = await adaptive.readGPUActivityPolicy();
        const activeMaximumFineCellX = Math.max(...activity.bricks.filter((brick) => brick.active)
          .map((brick) => 8 * (brick.coordinate[0] + 1) - 1));
        assert.ok(activeMaximumFineCellX >= final.adaptive.front.trace,
          `front x=${final.adaptive.front.trace} escaped active residency x=${activeMaximumFineCellX}`);
      } finally {
        adaptive.destroy();
        allFine.destroy();
      }
      const validation = await device.popErrorScope();
      assert.equal(validation?.message, undefined);
      assert.deepEqual(uncaptured, []);
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
