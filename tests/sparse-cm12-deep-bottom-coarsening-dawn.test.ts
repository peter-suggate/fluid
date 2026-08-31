import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import "../lib/methods";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { unpackFineLevelSetPackedFlags,
  unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import { resolveMethodValues } from "../lib/core/method-contract";
import { getScenePreset } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { parseQueryState } from "../lib/core/url-state";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

interface BottomBrickSample {
  readonly coordinate: readonly [number, number, number];
  readonly resolution: number;
  readonly reasons: number;
  readonly planReasons: number;
  readonly meanFineDensity: number;
  readonly minimumFineDensity: number;
}

interface StepSample {
  readonly step: number;
  readonly topologyGeneration: number;
  readonly bricks: readonly BottomBrickSample[];
  readonly verticalLadder: readonly {
    readonly resolution: number;
    readonly reasons: number;
    readonly planReasons: number;
  }[];
}

async function readWords(device: GPUDevice, source: GPUBuffer,
  words: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ size: Math.max(4, 4 * words),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, 4 * words);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function readPublishedTopHeights(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver): Promise<Float32Array> {
  const source = solver.globalFineLevelSetSource;
  const { plan } = source;
  const capacity = plan.maximumResidentBricks;
  const [worklist, metadata, samples] = await Promise.all([
    readWords(device, source.worklist, 7 + capacity),
    readWords(device, source.metadata, 4 * capacity),
    readWords(device, source.samples, plan.payloadCapacityBytes / 4),
  ]);
  const [nx, ny, nz] = plan.sampleDimensions;
  const r = plan.brickResolution;
  const field = new Float32Array(nx * ny * nz).fill(Number.NaN);
  for (let work = 0; work < worklist[1]!; work += 1) {
    const page = worklist[7 + work]!;
    const key = metadata[4 * page + 1]!;
    const bx = (key & 0x7ff) - 1024;
    const by = ((key >>> 11) & 0x3ff) - 512;
    const bz = ((key >>> 21) & 0x7ff) - 1024;
    for (let local = 0; local < plan.samplesPerBrick; local += 1) {
      const x = bx * r + local % r;
      const y = by * r + Math.floor(local / r) % r;
      const z = bz * r + Math.floor(local / (r * r));
      if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) continue;
      const packed = samples[page * plan.samplesPerBrick + local]!;
      if ((unpackFineLevelSetPackedFlags(packed) & 1) === 0) continue;
      field[x + nx * (y + ny * z)] = unpackFineLevelSetPackedPhi(packed);
    }
  }
  const heights = new Float32Array(nx * nz).fill(Number.NaN);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    for (let y = ny - 2; y >= 0; y -= 1) {
      const lower = field[x + nx * (y + ny * z)]!;
      const upper = field[x + nx * (y + 1 + ny * z)]!;
      if (!Number.isFinite(lower) || !Number.isFinite(upper)
        || lower >= 0 || upper < 0) continue;
      heights[x + nx * z] = y + 0.5
        +Math.max(0,Math.min(1,-lower/(upper-lower)));
      break;
    }
  }
  return heights;
}

dawnTest("Sparse CM12 publishes coarsening-biased hydrostatic ladders", {
  timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test",
    "tests/sparse-cm12-deep-bottom-coarsening-dawn.test.ts");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([
      `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
      "enable-dawn-features=disable_blob_cache",
    ]);
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: { gpu },
    });
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "Dawn must expose a WebGPU adapter");
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    const validationErrors: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      validationErrors.push(event.error.message);
    });

    const sourceScene = getScenePreset("water-box-dam-break").create();
    const scene = sceneAtContainerExtents(sourceScene, {
      width_m: sourceScene.container.width_m,
      height_m: 2,
      depth_m: sourceScene.container.depth_m,
    });
    // Preserve the production corner-dam shape in a taller tank and union in a
    // 26-cell-deep pool. Four vertical brick pages expose the surface-to-floor
    // ladder; activity mode deliberately biases its broad calm interface one
    // rung coarse, without violating strong 2:1 grading.
    scene.fluid.initialDamBreakDimensions_m = { x: 0.6, y: 1.8, z: 0.5 };
    scene.fluid.initialLiquidVolumes = [{
      shape: "box",
      min_m: { x: -0.5 * scene.container.width_m, y: 0,
        z: -0.5 * scene.container.depth_m },
      max_m: { x: 0.5 * scene.container.width_m, y: 1.3,
        z: 0.5 * scene.container.depth_m },
    }];
    const values = {
      ...adaptiveMassMethod.presetFor("balanced"),
      resolutionMode: "adaptive",
      brickFineResolution: "8",
      presentationPageResolution: "8",
      surfaceFineRings: 1,
      timeStep: "paper",
      secondaryParticles: "off",
    };
    solver = await adaptiveMassMethod.createSolverAsync!(
      device, scene, "balanced", values, undefined, () => {},
    ) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [24, 40, 16]);

    const samples: StepSample[] = [];
    const sample = async (step: number) => {
      await device!.queue.onSubmittedWorkDone();
      const [activity, fields] = await Promise.all([
        solver!.readGPUActivityPolicy(),
        solver!.readDiagnosticFields(),
      ]);
      const bottom = activity.bricks.filter((brick) => brick.active
        && brick.coordinate[1] === 0
        && brick.coordinate[0] >= 0 && brick.coordinate[0] < 3
        && brick.coordinate[2] >= 0 && brick.coordinate[2] < 2)
        .map<BottomBrickSample>((brick) => {
          let sum = 0, minimum = Infinity, count = 0;
          const x0 = brick.coordinate[0] * 8;
          const z0 = brick.coordinate[2] * 8;
          for (let z = z0; z < z0 + 8; z += 1) {
            for (let y = 0; y < 8; y += 1) {
              for (let x = x0; x < x0 + 8; x += 1) {
                const rho = fields.density[x + 24 * (y + 40 * z)]!;
                sum += rho;minimum = Math.min(minimum, rho);count += 1;
              }
            }
          }
          return {
            coordinate: brick.coordinate,
            resolution: brick.acceptedResolution,
            reasons: brick.reasons,
            planReasons: brick.planReasons,
            meanFineDensity: sum / count,
            minimumFineDensity: minimum,
          };
        }).sort((left, right) => left.coordinate[2] - right.coordinate[2]
          || left.coordinate[0] - right.coordinate[0]);
      assert.equal(bottom.length, 6,
        `the full-floor pool must keep all six bottom bricks resident on step ${step}`);
      const verticalLadder = Array.from({ length: 4 }, (_, y) => {
        const brick = activity.bricks.find((candidate) => candidate.active
          && candidate.coordinate[0] === 2 && candidate.coordinate[1] === y
          && candidate.coordinate[2] === 1);
        return { resolution: brick?.acceptedResolution ?? 0,
          reasons: brick?.reasons ?? 0, planReasons: brick?.planReasons ?? 0 };
      });
      samples.push({ step, topologyGeneration: activity.acceptedTopologyGeneration,
        bricks: bottom, verticalLadder });
    };

    await sample(0);
    assert.deepEqual(samples[0]!.verticalLadder.map((entry) => entry.resolution),
      [1, 1, 2, 4],
      "initial planar-wall restriction must omit B8 for a broad calm surface");
    // Two simulated seconds cover fifteen topology epochs and the dam impact,
    // long enough for the former B8/B4 ping-pong to complete several cycles.
    for (let step = 1; step <= 60; step += 1) {
      assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true,
        `advance ${step}`);
      await sample(step);
    }

    const failures: string[] = [];
    for (let brick = 0; brick < 6; brick += 1) {
      const history = samples.map((entry) => ({
        step: entry.step,
        generation: entry.topologyGeneration,
        ...entry.bricks[brick]!,
      }));
      const independentlyDeep = history.filter((entry) =>
        entry.meanFineDensity >= 0.95 && entry.minimumFineDensity >= 0.75);
      const label = history[0]!.coordinate.join(",");
      const profile = history.map((entry) =>
        `${entry.step}:${entry.resolution}/${entry.reasons}@${entry.generation}`).join(" ");
      // A genuinely fine interface two pages above may move its strong-2:1
      // support cone down by one rung. The floor itself must still remain B2
      // or coarser; B4/B8 here is the original blanket-boundary regression.
      const fine = independentlyDeep.filter((entry) => entry.resolution > 2);
      if (fine.length > 0) failures.push(
        `${label} fine while deep at [${fine.map((entry) => entry.step).join(",")}]; ${profile}`,
      );
      const falseSurface = independentlyDeep.filter((entry) => (entry.reasons & 1) !== 0);
      if (falseSurface.length > 0) failures.push(
        `${label} false surface at [${falseSurface.map((entry) => entry.step).join(",")}]; ${profile}`,
      );
    }
    assert.deepEqual(validationErrors, []);
    const ladderProfile = samples.map((entry) => entry.step + ":"
      + entry.verticalLadder.map((brick) => brick.resolution + "/"
        + brick.reasons + "/p" + brick.planReasons).join(",")).join(" ");
    assert.deepEqual(failures, [],
      `deep bottom coarsening was not stable:\n${failures.join("\n")}`
        + `\nvertical ladder ${ladderProfile}`);

    solver.destroy();
    solver = undefined;

    // Enter through the exact studio query and method-resolution path used by
    // the reported UI scene. This covers both the paused/reset grid frame and
    // the running policy; a hand-built solver-options approximation previously
    // missed the product thresholds and did not catch the unchanged reset view.
    const offsetUI = parseQueryState(
      "?scene=hydrostatic-power-large-offset&method=adaptive-mass&grid=volume",
    );
    assert.equal(offsetUI.methodId, "adaptive-mass");
    assert.equal(offsetUI.ui.gridOverlayAxis, "volume");
    assert.equal(offsetUI.ui.gridOverlayMode, "structure");
    const offsetValues = resolveMethodValues(adaptiveMassMethod,
      offsetUI.quality, offsetUI.overrides[offsetUI.methodId] ?? {});
    assert.equal(offsetValues.selectorMode, "activity");
    assert.equal(offsetValues.resolutionMode, "adaptive");
    const offsetSolver = await adaptiveMassMethod.createSolverAsync!(
      device, offsetUI.scene, offsetUI.quality, offsetValues, undefined,
      () => {},
    ) as WebGPUAdaptiveMassSolver;
    try {
      await offsetSolver.waitForSimulationReady();
      const resetSnapshot = await offsetSolver.readGPUActivityPolicy();
      const resetSurface = resetSnapshot.bricks.filter((brick) => brick.active
        && brick.coordinate[1] === 1);
      assert.equal(resetSurface.length, 8,
        "the exact UI reset frame must contain all eight surface pages");
      assert.ok(resetSurface.every((brick) => brick.acceptedResolution === 4),
        `the exact UI reset frame must present its calm surface at B4: ${
          resetSurface.map((brick) => `${brick.coordinate.join(",")}=${
            brick.acceptedResolution}`).join("; ")}`);
      assert.ok(resetSurface.every((brick) => (brick.reasons & 64) !== 0),
        "the exact UI reset surface must publish occupied pressure topology");
      const resetHeights = await readPublishedTopHeights(device, offsetSolver);
      const resetFinite = [...resetHeights].filter(Number.isFinite);
      assert.equal(resetFinite.length, 32 * 16,
        "the exact UI reset must publish every large-offset surface column");
      const resetMean = resetFinite.reduce((sum, height) => sum + height, 0)
        /resetFinite.length;
      assert.ok(Math.abs(resetMean - 15.25) <= 0.01,
        `large-offset reset waterline was ${resetMean}, expected 15.25 cells`);
      let settledGeneration: number | undefined;
      for (let step = 1; step <= 16; step += 1) {
        assert.equal(offsetSolver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const snapshot = await offsetSolver.readGPUActivityPolicy();
        const surfaceLayer = snapshot.bricks.filter((brick) => brick.active
          && brick.coordinate[1] === 1);
        const drySupportLayer = snapshot.bricks.filter((brick) => brick.active
          && brick.coordinate[1] === 2);
        assert.equal(surfaceLayer.length, 8,
          `large-offset surface layer changed membership at step ${step}`);
        assert.ok(surfaceLayer.every((brick) => brick.acceptedResolution === 4
          && brick.plannedResolution === 4),
        `large-offset surface did not retain B4 at step ${step}: ${
          surfaceLayer.map((brick) => `${brick.coordinate.join(",")}=${
            brick.acceptedResolution}/${brick.plannedResolution}/p${
            brick.planReasons}/r${brick.reasons}/s${brick.scoreByte}`).join("; ")}`);
        assert.equal(drySupportLayer.length, 8,
          `large-offset dry support layer changed membership at step ${step}`);
        assert.ok(drySupportLayer.every((brick) => (brick.reasons & 64) === 0),
          `large-offset support became liquid topology at step ${step}: ${
            drySupportLayer.map((brick) => `${brick.coordinate.join(",")}=${
              brick.acceptedResolution}/${brick.plannedResolution}/p${
              brick.planReasons}/r${brick.reasons}/s${brick.scoreByte}`).join("; ")}`);
        if (step === 1) {
          settledGeneration = snapshot.acceptedTopologyGeneration;
        } else {
          assert.equal(snapshot.acceptedTopologyGeneration, settledGeneration,
            `large-offset calm surface churned topology at step ${step}`);
        }
        assert.equal(snapshot.commitFailed, false);
        if (step === 1) {
          const steppedHeights = await readPublishedTopHeights(device, offsetSolver);
          let compared = 0, maximumChange = 0;
          for (let column = 0; column < resetHeights.length; column += 1) {
            const before = resetHeights[column]!, after = steppedHeights[column]!;
            if (!Number.isFinite(before) || !Number.isFinite(after)) continue;
            compared += 1;maximumChange = Math.max(maximumChange,
              Math.abs(after - before));
          }
          assert.equal(compared, resetFinite.length);
          assert.ok(maximumChange <= 0.02,
            `large-offset first step moved a surface column by ${maximumChange} cells`);
        }
      }
    } finally {
      offsetSolver.destroy();
    }
    assert.deepEqual(validationErrors, []);
  } finally {
    solver?.destroy();
    device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
});
