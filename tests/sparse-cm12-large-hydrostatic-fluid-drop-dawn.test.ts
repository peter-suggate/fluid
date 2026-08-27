import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { defaultFluidBallRadius_m, fluidInteractionDropVolume } from
  "../lib/core/editor-fluid-volume";
import { hoverSceneAt, restFluidInWorld } from "../lib/core/editor-hover";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDefinitionCamera, sceneDocument } from "../lib/core/scene-definition";
import { findSceneDefinition } from "../lib/core/scenes";
import { projectToViewport, viewportRayForPointer } from "../lib/core/webgpu-camera";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
const sum = (values: Float32Array) => values.reduce((total, value) => total + value, 0);

dawnTest("Sparse CM12 accepts a UI-positioned drop in the larger hydrostatic scene",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-large-hydrostatic-fluid-drop-dawn.test.ts");
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
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const definition = findSceneDefinition("hydrostatic-power-large-offset")!;
      const scene = sceneDocument(definition);
      const radius_m = defaultFluidBallRadius_m(scene);
      const target = {
        x: 0,
        y: scene.container.fillFraction * scene.container.height_m,
        z: 0,
      };
      const camera = sceneDefinitionCamera(definition);
      const viewport = { left: 0, top: 0, width: 1542, height: 784 };
      const projected = projectToViewport(target, camera, viewport.width, viewport.height);
      const ray = viewportRayForPointer(camera,
        projected.leftFraction * viewport.width,
        projected.topFraction * viewport.height,
        viewport);
      const centre = restFluidInWorld(scene, ray, hoverSceneAt(scene, [], ray), radius_m);
      assert.ok(centre, "the visible free surface must accept the drop gesture");
      const drop = fluidInteractionDropVolume(scene, centre, radius_m);

      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true,
        "the UI run must advance before taking the live-injection path");
      await device.queue.onSubmittedWorkDone();

      const beforeMass = sum((await solver.readDiagnosticFields()).density);
      const beforeGeneration = solver.sparseWorld.status().acceptedGeneration;
      solver.injectLiquidBall({ centre_m: drop.center_m, radius_m: drop.radius_m });
      assert.equal(solver.sparseWorld.status().acceptedGeneration, beforeGeneration + 1,
        "the UI drop must publish exactly one sparse-world generation");
      await device.queue.onSubmittedWorkDone();

      const afterMass = sum((await solver.readDiagnosticFields()).density);
      const growth = await solver.readWorldGrowthReceiptQA();
      const activity = await solver.readGPUActivityPolicy();
      const representedMass = afterMass + growth.dynamicLiquidMassFineCells;
      assert.ok(representedMass > beforeMass + 1,
        `the UI-positioned drop added no visible liquid (${beforeMass} -> ${representedMass}); `
        + `drop=${JSON.stringify(drop)}, `
        + `active=${activity.residentBrickCount}, new=${activity.newlyActivatedBrickCount}, `
        + `faults=${activity.faultFlags}, growth=${JSON.stringify(growth)}`);

      assert.equal(solver.advanceTo(2 * CM12_PAPER_DT_S, []), true,
        "the injected pages must remain valid on the next physics step");
      await device.queue.onSubmittedWorkDone();
      const settledMass = sum((await solver.readDiagnosticFields()).density)
        + (await solver.readWorldGrowthReceiptQA()).dynamicLiquidMassFineCells;
      assert.ok(settledMass > beforeMass + 1,
        `the next step exposed gaps in the injected pages (${representedMass} -> ${settledMass})`);

      const outside = fluidInteractionDropVolume(scene, {
        x: -scene.container.width_m / 2 - 2 * radius_m,
        y: scene.container.height_m / 2,
        z: 0,
      }, radius_m);
      const outsideGeneration = solver.sparseWorld.status().acceptedGeneration;
      solver.injectLiquidBall({ centre_m: outside.center_m, radius_m: outside.radius_m });
      await device.queue.onSubmittedWorkDone();
      assert.equal(solver.sparseWorld.status().acceptedGeneration, outsideGeneration + 1);
      const outsideMass = sum((await solver.readDiagnosticFields()).density)
        + (await solver.readWorldGrowthReceiptQA()).dynamicLiquidMassFineCells;
      assert.ok(outsideMass > settledMass + 1,
        `an open-world drop outside the tank added no liquid (${settledMass} -> ${outsideMass})`);
      assert.ok((await solver.readWorldGrowthReceiptQA()).minimum[0] < 0,
        "the outside drop must publish a signed page beyond the original tank");

      assert.equal(solver.advanceTo(3 * CM12_PAPER_DT_S, []), true,
        "the outside-tank drop must survive its following physics step");
      await device.queue.onSubmittedWorkDone();
      const outsideSettledMass = sum((await solver.readDiagnosticFields()).density)
        + (await solver.readWorldGrowthReceiptQA()).dynamicLiquidMassFineCells;
      assert.ok(outsideSettledMass > settledMass + 1,
        `the outside drop vanished on the next step (${outsideMass} -> ${outsideSettledMass})`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
