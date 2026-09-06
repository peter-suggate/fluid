import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { SparseCM12GenerationPlanningGate } from "../lib/methods/adaptive-mass/sparse-cm12-generation-planning-gate";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("generation preflight preserves rerungs, activation, macro motion and quiet merges", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "generation-planning-gate");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  let gate: SparseCM12GenerationPlanningGate | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const count = 16, header = 32, stride = 48;
    const activity = device.createBuffer({ size: 4 * (header + count * stride),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const metadata = new Uint32Array(count).fill(1 << 8);
    metadata[1] |= 1; // This leaf already has an in-place candidate.
    metadata[2] = 4 << 8; // Physical macro.
    gate = await SparseCM12GenerationPlanningGate.create(device, activity, metadata, header, stride);
    const words = new Uint32Array(header + count * stride), floats = new Float32Array(words.buffer);
    const reset = () => {
      words.fill(0);
      for (let i = 0; i < count; i++) { words[header + stride * i + 10] = 1; words[header + stride * i + 12] = 1; }
    };
    const check = async (expected: boolean, label: string, maximumSpan = 8) => {
      device!.queue.writeBuffer(activity, 0, words);
      assert.equal(await gate!.needed(maximumSpan, 64, 1), expected, label);
    };
    reset(); await check(false, "calm leaves before merge persistence need no CPU snapshot");
    words[header + 47] = 8; await check(true, "unbacked refinement");
    words[header + 10] = 0; await check(false, "inactive undemanded leaf");
    words[header + 9] = 0x80000000; await check(true, "demanded inactive activation");
    reset(); words[header + stride + 47] = 8; await check(false, "backed work stays on GPU");
    reset(); floats[header + 2 * stride + 33] = 1; await check(true, "macro velocity demand");
    reset(); words[header + 2 * stride + 1] = 256; await check(true, "thin macro demand");
    reset();
    for (let i = 0; i < 8; i++) {
      words[header + i * stride + 2] = 64 << 8; floats[header + i * stride + 4] = 1;
      await check(i === 7, `${i + 1} potential merge siblings`);
    }
    await check(false, "physical maximum span forbids all merges", 1);
    words[header + 1] = 1; await check(false, "surface witness vetoes one merge sibling");
    activity.destroy();
  } finally { gate?.destroy(); device?.destroy(); releaseWebGPUExclusiveLock(); void gpu; }
});
