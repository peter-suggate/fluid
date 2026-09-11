import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("Sparse CM12 couples the settled-tank rigid bodies without losing water",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-rigid-coupling-dawn.test.ts");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn,
        [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      const scene = sceneDocument(getSceneDefinition("water-box-tank-fill"));
      const bodies = initializeRigidBodies(scene.rigidBodies);
      assert.equal(bodies.length, 2, "the acceptance scene must retain both authored bodies");
      const authoredY = bodies.map((body) => body.position_m.y);
      device.pushErrorScope("validation");
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined,
        sparseCM12DawnDefaultOptions(),
        () => {},
      );
      try {
        await solver.waitForSimulationReady();
        const initialMass = solver.info.volumeCellSum!;
        const assertHealthyWithVolumeQA = async () => {
          try {
            await solver.assertSimulationHealthy();
          } catch (error) {
            const owner = /owner=(\d+)/.exec(String(error));
            console.error("geometric-rigid-failure", JSON.stringify({
              transport: await solver.readGeometricVolumeTransportReceiptQA(),
              cell: owner ? await solver.readAcceptedGeometricCellRowsQA(Number(owner[1])) : undefined,
            }, null, 2));
            throw error;
          }
        };
        for (let step = 1; step <= 30; step += 1) {
          let nextAdmissionDiagnostic = performance.now() + 15_000;
          while (!solver.advanceTo(step * CM12_PAPER_DT_S, bodies)) {
            const deferred = solver.info.topologyGenerationDeferred;
            if (deferred?.reason === "volume-capacity") {
              throw new Error(`Rigid step ${step} requires topology admission: ${deferred.detail ?? JSON.stringify(deferred)}`);
            }
            if (performance.now() >= nextAdmissionDiagnostic) {
              console.error("geometric-rigid-admission-wait", JSON.stringify({
                step, framePending: solver.framePending,
                topologyGenerationPending: solver.info.topologyGenerationPending,
                topologyGenerationDeferred: deferred,
                topologyGenerationError: solver.info.topologyGenerationError,
              }));
              nextAdmissionDiagnostic = performance.now() + 15_000;
            }
            await new Promise(setImmediate);
          }
          try { await solver.awaitFrameCompletion(); }
          catch (error) { await assertHealthyWithVolumeQA(); throw error; }
          if (process.env.FLUID_GEOMETRIC_STEP_QA === "1") {
            await device.queue.onSubmittedWorkDone();
            const transport = await solver.readGeometricVolumeTransportReceiptQA();
            console.error("geometric-rigid-step", JSON.stringify({ step, transport,
              physical: await solver.readAcceptedGeometricVolumeQA(),
            }));
            await assertHealthyWithVolumeQA();
          }
        }
        await device.queue.onSubmittedWorkDone();
        await assertHealthyWithVolumeQA();

        const poses = await solver.readRigidBodyPoses();
        assert.equal(poses?.length, bodies.length);
        poses!.forEach((pose, index) => {
          assert.ok(Number.isFinite(pose.position_m.x));
          assert.ok(Number.isFinite(pose.position_m.y));
          assert.ok(Number.isFinite(pose.position_m.z));
          assert.ok(pose.position_m.y < authoredY[index]! - 0.25,
            `body ${index} did not enter the sparse fluid domain: authored y=${
              authoredY[index]}, published y=${pose.position_m.y}`);
        });
        assert.ok(poses![0]!.position_m.y > poses![1]!.position_m.y + 0.05,
          "the cork should remain above the dense box after entering the water");

        const fields = await solver.readDiagnosticFields();
        let mass = 0;
        for (let index = 0; index < fields.density.length; index += 1) {
          assert.ok(Number.isFinite(fields.density[index]));
          assert.ok(Number.isFinite(fields.velocity[index]));
          assert.ok(Number.isFinite(fields.pressure[index]));
          assert.ok(Number.isFinite(fields.divergence[index]));
          mass += fields.density[index]!;
        }
        // One finest-cell unit is a 0.065% bound here and includes the sparse
        // scheduler's deliberate sub-residency residue retirement.
        assert.ok(Math.abs(mass - initialMass) <= 1,
          `moving-solid coverage changed mass by ${mass - initialMass} fine cells`);
      } finally {
        solver.destroy();
      }
      const validation = await device.popErrorScope();
      assert.equal(validation?.message, undefined);
      assert.deepEqual(uncaptured, []);
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
