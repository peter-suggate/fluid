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

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

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
      const createSolver = (resolutionMode: "adaptive" | "all-fine") =>
        WebGPUAdaptiveMassSolver.createAsync(device!, scene, "balanced", undefined, {
          resolutionMode,
          fineTileResolution: 8,
          coarseTileResolution: 4,
          timeStep: "paper",
        }, () => {});
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
          const activeBricks = activity.bricks.filter((brick) => brick.active);
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
