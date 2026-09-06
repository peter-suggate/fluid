import assert from "node:assert/strict";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { mkdir, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// Retain the native Dawn instance until all asynchronous readbacks finish.
const live = new Set<GPU>();
async function read(device: GPUDevice, source: GPUBuffer, bytes = source.size, offset = 0) {
  const target = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, offset, target, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await target.mapAsync(GPUMapMode.READ);
    return new Uint8Array(target.getMappedRange()).slice();
  } finally {
    if (target.mapState === "mapped") target.unmap();
    target.destroy();
  }
}

const dawnModule = process.env.WEBGPU_NODE_MODULE;
assert.ok(dawnModule, "Set WEBGPU_NODE_MODULE to the native Dawn module path");
await acquireWebGPUExclusiveLock("dawn-probe", "pool-reset-surface");
const errors: string[] = [];
let gpu: GPU | undefined;
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const { create, globals } = await import(pathToFileURL(dawnModule).href);
  Object.assign(globalThis, globals);
  gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]) as GPU;
  live.add(gpu);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device.addEventListener("uncapturederror", event => errors.push(event.error.message));
  const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half"));
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", { selectorMode: "coarse-first", timeStep: "scene" });
  solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => { }) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  {
    const step = 0;
    const output = `${process.env.RESET_OUTPUT ?? "artifacts/pool-reset-surface"}/step-${step}`;
    await mkdir(output, { recursive: true });
    const stats = await solver.readStats();
    assert.ok(Math.abs((stats.simulatedTime_s ?? NaN) - step * CM12_PAPER_DT_S) < 1e-9);
    const fields = await solver.readDiagnosticFields(true);
    for (const name of ["density", "solidOpenFraction", "velocity", "pressure"] as const)
      await writeFile(`${output}/${name}.bin`, new Uint8Array(fields[name].buffer));
    const source: WebGPUAdaptiveMassSolver["globalFineLevelSetSource"] = solver.globalFineLevelSetSource;
    const blobs: Uint8Array[] = await Promise.all([
      read(device, source.worklist), read(device, source.metadata),
      read(device, source.samples, source.plan.payloadCapacityBytes),
    ]);
    const [worklist, metadata, samples] = blobs.map(b => new Uint32Array(b.buffer));
    const names = ["worklist", "metadata", "samples"];
    for (let i = 0; i < 3; i++)
      await writeFile(`${output}/${names[i]}.bin`, blobs[i]!);
    await writeFile(`${output}/source.json`, JSON.stringify(source.plan));
    const [nx, ny, nz] = source.plan.sampleDimensions;
    assert.equal(source.plan.brickResolution, 8);
    // This dense convenience view covers ordinary pages around the pool's
    // free surface. Leave macro interiors as NaN; their complete encoded data
    // remains in metadata.bin / samples.bin for other consumers.
    const phi = new Float32Array(nx * ny * nz).fill(NaN), width = new Uint8Array(nx * ny * nz);
    for (let at = 0; at < worklist![1]!; at++) {
      const page = worklist![7 + at]!, key = metadata![4 * page + 1]!;
      assert.equal(metadata![4 * page], page);
      assert.equal(metadata![4 * page + 2], worklist![0]);
      const descriptor = metadata![4 * page + 3]!;
      if ((descriptor & 0x80000000) !== 0 && ((descriptor >>> 24) & 31) !== 0) continue;
      const bx = (key & 2047) - 1024, by = ((key >>> 11) & 1023) - 512, bz = ((key >>> 21) & 2047) - 1024;
      for (let q = 0; q < 512; q++) {
        const x = bx * 8 + q % 8, y = by * 8 + Math.floor(q / 8) % 8, z = bz * 8 + Math.floor(q / 64);
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
          continue;
        const packed = samples![page * 512 + q]!;
        if (!(packed & 0x10000))
          continue;
        phi[x + nx * (y + ny * z)] = unpackFineLevelSetPackedPhi(packed);
        width[x + nx * (y + ny * z)] = 1 << ((packed >>> 24) & 15);
      }
    }
    await writeFile(`${output}/phi.bin`, new Uint8Array(phi.buffer));
    await writeFile(`${output}/width.bin`, width);
    await writeFile(`${output}/activity.json`, JSON.stringify(await solver.readGPUActivityPolicy()));
    await writeFile(`${output}/stats.json`, JSON.stringify(await solver.readStats()));
    console.log(JSON.stringify({ step, time: step * CM12_PAPER_DT_S, output, errors }));
  }
  assert.deepEqual(errors, []);
} finally {
  solver?.destroy();
  device?.destroy();
  await releaseWebGPUExclusiveLock();
  if (gpu) live.delete(gpu);
}
