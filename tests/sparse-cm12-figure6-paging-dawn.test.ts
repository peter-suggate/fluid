import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createCm12Figure6 } from "../lib/core/cm12-paper-scenes";
import {
  unpackFineLevelSetPackedFlags,
  unpackFineLevelSetPackedPhi,
} from "../lib/core/fine-levelset-packed-sample";
import { resolveMethodValues } from "../lib/core/method-contract";
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

async function readNegativeRows(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver): Promise<ReadonlySet<number>> {
  const source = solver.globalFineLevelSetSource;
  const { plan } = source;
  const capacity = plan.maximumResidentBricks;
  const [worklist, metadata, samples] = await Promise.all([
    readWords(device, source.worklist, 7 + capacity),
    readWords(device, source.metadata, 4 * capacity),
    readWords(device, source.samples, plan.payloadCapacityBytes / 4),
  ]);
  const rows = new Set<number>();
  const resolution = plan.brickResolution;
  for (let work = 0; work < worklist[1]!; work += 1) {
    const page = worklist[7 + work]!;
    const key = metadata[4 * page + 1]!;
    const brickY = ((key >>> 11) & 0x3ff) - 512;
    for (let local = 0; local < plan.samplesPerBrick; local += 1) {
      const packed = samples[page * plan.samplesPerBrick + local]!;
      if ((unpackFineLevelSetPackedFlags(packed) & 1) === 0
        || unpackFineLevelSetPackedPhi(packed) >= 0) continue;
      rows.add(brickY * resolution
        + Math.floor(local / resolution) % resolution);
    }
  }
  return rows;
}

dawnTest("Figure 6 crosses its authored SparseWorld boundary",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-figure6-paging-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    const uncaptured: string[] = [];
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
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault(); uncaptured.push(event.error.message);
      });
      device.pushErrorScope("validation");

      const scene = createCm12Figure6();
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        brickFineResolution: "8", presentationPageResolution: "8",
        resolutionMode: "adaptive", selectorMode: "activity",
        surfaceFineRings: 1, timeStep: "paper",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
        values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      const [initialRows, initialStats] = await Promise.all([
        readNegativeRows(device, solver), solver.readStats(),
      ]);
      const initialMaximumY = Math.max(...initialRows);
      assert.ok((initialStats.adaptiveFineBrickCount ?? 0) > 0,
        "the curved drop must retain B8 roots for SparseWorld frontier growth");

      for (let step = 1; step <= 30; step += 1) {
        while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) {
          await new Promise(setImmediate);
        }
      }
      await device.queue.onSubmittedWorkDone();
      const [rows, growth, indirect, activity] = await Promise.all([
        readNegativeRows(device, solver), solver.readWorldGrowthReceiptQA(),
        solver.readAcceptedIndirectQA(), solver.readGPUActivityPolicy(),
      ]);
      const maximumY = Math.max(...rows);
      const receipt = { initialMaximumY, maximumY,
        publishedTopologyPages: growth.publishedTopologyPages,
        insertionFaults: growth.insertionFaults,
        capacityFaults: growth.capacityFaults,
        topologyFailed: activity.commitFailed,
        cellWorkgroups: indirect[0], rowWorkgroups: indirect[3],
        crossedRows: Array.from({ length: 16 }, (_, index) => index + 8)
          .filter((row) => rows.has(row)) };
      if (process.env.FLUID_FIGURE6_TRACE === "1") {
        process.stderr.write(`[cm12-figure6] ${JSON.stringify(receipt)}\n`);
      }
      assert.ok(growth.publishedTopologyPages > 512,
        `the drop must cross the former undersized page-pool ceiling: ${JSON.stringify(receipt)}`);
      assert.ok(indirect[0]! > 0 && indirect[3]! > 0, JSON.stringify(receipt));
      assert.ok(maximumY < initialMaximumY - 20, JSON.stringify(receipt));
      assert.deepEqual(receipt.crossedRows,
        Array.from({ length: 16 }, (_, index) => index + 8), JSON.stringify(receipt));
      assert.equal(growth.insertionFaults, 0, JSON.stringify(receipt));
      assert.equal(growth.capacityFaults, 0, JSON.stringify(receipt));
      assert.equal(activity.commitFailed, false, JSON.stringify(receipt));
      const validation = await device.popErrorScope();
      assert.equal(validation, null, validation?.message);
      assert.deepEqual(uncaptured, []);
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
