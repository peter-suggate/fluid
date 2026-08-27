import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
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

async function readGPUWords(device: GPUDevice, source: GPUBuffer,
  count: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ size: Math.max(4, 4 * count),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, 4 * count);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

dawnTest("mini32 presentation advances with the simulation through 0.467 s",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-mini32-presentation-progress-dawn.test.ts");
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
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();validationErrors.push(event.error.message);
      });

      const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        brickFineResolution: "8", resolutionMode: "adaptive",
        selectorMode: "surface", surfaceFineRings: 1, timeStep: "paper",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();

      const initialFrame = await solver.readFrameControlQA();
      const timeline: Array<Record<string, unknown>> = [];
      for (let step = 1; step <= 14; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        // Match the browser's two-deep submission queue. A per-step fence can
        // hide a presentation generation overwrite that occurs in production.
        if (step % 2 !== 0 && step !== 14) continue;
        await device.queue.onSubmittedWorkDone();
        const [frame, activity, fpp] = await Promise.all([
          solver.readFrameControlQA(), solver.readGPUActivityPolicy(),
          solver.readFramePlanPresentationHeaderQA(),
        ]);
        const fppFaultRecord = (fpp as Record<string, number>).faultCode !== 0
          ? await solver.readFramePlanPresentationFaultRecordQA() : undefined;
        const status = solver.sparseWorld.status();
        const presentation = solver.sparseWorld.presentation();
        const source = presentation.fineLevelSet;
        const directory = await readGPUWords(device, source.worklist,
          7 + source.plan.maximumResidentBricks);
        const residentPages = directory[1]!;
        const pages = directory.slice(7, 7 + residentPages);
        const metadata = await readGPUWords(device, source.metadata,
          4 * source.plan.maximumResidentBricks);
        const faultPage = (fppFaultRecord as { page?: number } | undefined)?.page;
        const faultPageMetadata = faultPage !== undefined
          && faultPage < source.plan.maximumResidentBricks
          ? Array.from(metadata.slice(4 * faultPage, 4 * faultPage + 4)) : undefined;
        const metadataGenerations = [...new Set(Array.from(pages,
          (page) => metadata[4 * page + 2]!))].sort((a, b) => a - b);
        timeline.push({ step, time_s: step * CM12_PAPER_DT_S,
          frame, topologyGeneration: activity.acceptedTopologyGeneration,
          topologyCommitFailed: activity.commitFailed,
          worldAcceptedGeneration: status.acceptedGeneration,
          worldLastAcceptedTime: status.lastAcceptedTime,
          presentationAcceptedGeneration: presentation.acceptedGeneration,
          sourceGeneration: source.generation,
          worklistHeader: Array.from(directory.slice(0, 7)),
          worklistGeneration: directory[0]! & 0x3fff_ffff,
          residentPages, metadataGenerations, fpp, fppFaultRecord,
          faultPageMetadata });
      }

      const issues: string[] = [];
      let priorFrameGeneration = initialFrame.acceptedGeneration;
      let priorPresentationGeneration = 0;
      for (const point of timeline) {
        const frame = point.frame as Awaited<ReturnType<
          WebGPUAdaptiveMassSolver["readFrameControlQA"]>>;
        const fpp = point.fpp as Record<string, number>;
        const step = point.step as number;
        const presentationGeneration = point.presentationAcceptedGeneration as number;
        const worklistGeneration = point.worklistGeneration as number;
        if (frame.acceptedGeneration <= priorFrameGeneration) {
          issues.push(`frame control stalled at step ${step}`);
        }
        if (presentationGeneration <= priorPresentationGeneration) {
          issues.push(`presentation object stalled at step ${step}`);
        }
        if (fpp.faultCode !== 0 || fpp.coverageFaultCount !== 0) {
          issues.push(`FPP1 fault ${fpp.faultCode}/${fpp.coverageFaultCount} at step ${step}`);
        }
        if (frame.acceptedGeneration !== initialFrame.acceptedGeneration + step
          || frame.committedFrames !== initialFrame.committedFrames + step
          || fpp.acceptedGeneration !== step || fpp.generationReceipt !== step
          || fpp.topologyGeneration !== point.topologyGeneration
          || point.worldAcceptedGeneration !== presentationGeneration
          || (point.metadataGenerations as number[]).some((generation) => generation !== 1)) {
          issues.push(`presentation generation diverged at step ${step}`);
        }
        priorFrameGeneration = frame.acceptedGeneration;
        priorPresentationGeneration = presentationGeneration;
      }
      if (process.env.FLUID_MINI32_PRESENTATION_TRACE === "1" || issues.length > 0) {
        process.stderr.write(`[mini32-presentation] ${JSON.stringify({
          initialFrame, timeline, validationErrors, issues,
        })}\n`);
      }
      assert.deepEqual(validationErrors, []);
      assert.deepEqual(issues, [], JSON.stringify(timeline));
    } finally {
      solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();
    }
  });
