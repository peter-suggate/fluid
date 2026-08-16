import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const brickId = (coordinate: readonly number[]) => coordinate.join(",");

test("Sparse CM12 advance does not make topology decisions from host readback", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  const start = source.indexOf("  advanceTo(time_s:");
  const end = source.indexOf("\n  private finishFrameCapture(", start);
  assert.ok(start >= 0 && end > start, "advanceTo source range must remain identifiable");
  const advance = source.slice(start, end);
  assert.doesNotMatch(advance,
    /mapAsync|onSubmittedWorkDone|readActivitySnapshot|readGPUActivityPolicy|readStats/,
    "GPU topology publication must not put a readback or queue fence in advanceTo");
});

dawnTest("Dawn publishes coarse-to-fine and fine-to-coarse Sparse CM12 topology",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-adaptive-resolution-lifecycle-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
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

      const scene = createMinimalPowerDamBreak64Scene();
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined,
        {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          // Keep this lifecycle fixture on the bounded compatibility path.
          // Large domains intentionally skip host-built resolution variants;
          // Figure 9 covers that accepted-only startup path separately.
          receiverSupportRings: 1,
          receiverFloor: 4,
          timeStep: "scene",
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            activitySignals: false,
            topologyCadenceSteps: 1,
            promoteEpochs: 1,
            demoteEpochs: 1,
          },
        },
        () => {},
      );

      const initial = await solver.readGPUActivityPolicy();
      const initialAccepted = new Map(initial.bricks.map((brick) =>
        [brickId(brick.coordinate), brick.acceptedResolution] as const));
      assert.ok(initial.bricks.some((brick) => brick.active
        && brick.acceptedResolution < 8),
      "the regression needs an already-created coarse reservoir/front brick");
      assert.ok(initial.bricks.some((brick) => brick.acceptedResolution === 8),
        "the construction band must seed physical fine support regions for demotion");
      const initialStats = await solver.readStats();
      const initialGeneration = initialStats.fluidBrickGeneration ?? 0;

      let promoted: (typeof initial.bricks)[number] | undefined;
      let coarsened: (typeof initial.bricks)[number] | undefined;
      let preparedPromotion: (typeof initial.bricks)[number] | undefined;
      let preparedCoarsening: (typeof initial.bricks)[number] | undefined;
      const dt_s = scene.numerics.maxDt_s;
      assert.equal(dt_s, 0.004, "scene-step mode must exercise the authored 4 ms step");
      for (let step = 1; step <= 180 && (!promoted || !coarsened); step += 1) {
        assert.equal(solver.advanceTo(step * dt_s, []), true);
        assert.ok(Math.abs((solver.info.lastDt_s ?? 0) - dt_s) < 1e-12,
          `advance ${step} must remain a 4 ms scene step`);
        // These explicit QA snapshots occur after an accepted advance. They are
        // test observations and never size or select production dispatches.
        const activity = await solver.readGPUActivityPolicy();
        promoted ??= activity.bricks.find((brick) =>
          initialAccepted.get(brickId(brick.coordinate))! < 8
            && brick.acceptedResolution === 8
            && (brick.reasons & 1) !== 0);
        preparedPromotion ??= activity.bricks.find((brick) =>
          initialAccepted.get(brickId(brick.coordinate))! < 8
            && brick.candidateResolution === 8
            && brick.transferStatus === 1
            && brick.faceTransferStatus === 1);
        coarsened ??= activity.bricks.find((brick) =>
          initialAccepted.get(brickId(brick.coordinate)) === 8
            && (brick.reasons & 1) === 0
            && brick.acceptedResolution < 8);
        preparedCoarsening ??= activity.bricks.find((brick) =>
          initialAccepted.get(brickId(brick.coordinate)) === 8
            && (brick.reasons & 1) === 0
            && brick.candidateResolution < 8
            && brick.transferStatus === 1
            && brick.faceTransferStatus === 1);
      }

      assert.ok(promoted,
        `moving free-surface fluid must publish an existing coarse brick as physical 8^3; `
          + `prepared candidate=${preparedPromotion?.candidateResolution ?? "none"}, `
          + `accepted=${preparedPromotion?.acceptedResolution ?? "none"}; `
          + `quiet candidate=${preparedCoarsening?.candidateResolution ?? "none"}, `
          + `accepted=${preparedCoarsening?.acceptedResolution ?? "none"}`);
      assert.equal(promoted.acceptedResolution, 8,
        "a planned/candidate 8^3 request is not an accepted physical split");
      assert.ok(coarsened,
        "an initially fine brick that no longer intersects the surface must publish a coarser "
          + `rung; prepared candidate=${preparedCoarsening?.candidateResolution ?? "none"}, `
          + `accepted=${preparedCoarsening?.acceptedResolution ?? "none"}`);
      assert.ok(coarsened.acceptedResolution === 4
        || coarsened.acceptedResolution === 2 || coarsened.acceptedResolution === 1);

      const stats = await solver.readStats();
      assert.ok((stats.fluidBrickGeneration ?? 0) > initialGeneration,
        "accepted topology publication must advance the resident generation");
      assert.equal(solver.info.hostSchedulingUsesReadback, false);
      assert.equal(solver.info.hostSimulationSizedWorkItems, 0);
      assert.deepEqual(uncaptured, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      releaseWebGPUExclusiveLock();
    }
  });
