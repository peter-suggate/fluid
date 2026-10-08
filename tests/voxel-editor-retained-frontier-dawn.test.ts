import { readFileSync } from "node:fs";
import { parseScene } from "../lib/core/model";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import {createUniformSolver,readUniformFields} from "./helpers/uniform-geometric";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";


const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("Uniform mixed ownership advances the edited scene without replacing its fields", { timeout: 240000 }, async () => {
  let device: GPUDevice | undefined;
  let solver: WebGPUUniformReferenceSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", (event) => { event.preventDefault(); errors.push(event.error.message); });
    const scene = parseScene(readFileSync(new URL("./fixtures/voxel-editor-live-scene.json", import.meta.url), "utf8"));
    device = managedGPUDevice(device,{requireWorkerRealm:false});
    const solverScene=scene;solverScene.numerics.fixedDt_s=solverScene.numerics.maxDt_s=1/30;
    solver = await createUniformSolver(device,solverScene);
    const dt = 1 / 30;
    const advance = async (time: number) => {
      while (!solver!.advanceTo(time, [])) await new Promise(setImmediate);
      await solver!.awaitFrameCompletion?.();
      await device!.queue.onSubmittedWorkDone();
    };
    await advance(dt);
    const world = solver.volumeTexture;
    for (let frame = 2; frame <= 90; frame++) {
      await advance(frame * dt);
      if (frame % 10 === 0 || frame === 29) {
        await readUniformFields(device,solver);
        console.log(JSON.stringify({ frame, time: solver.info.submittedTime_s }));
      }
    }
    const final = await readUniformFields(device,solver);
    assert.ok(final.density.every(Number.isFinite));
    assert.ok(final.pressure.every(Number.isFinite));
    assert.equal(solver.volumeTexture, world);
    assert.deepEqual(errors, []);
    assert.ok(solver.info.submittedTime_s! >= 3 - 1e-8);
  } finally {
    solver?.destroy();
    if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
  }
});
