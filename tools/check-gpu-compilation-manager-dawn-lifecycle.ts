/**
 * Minimal native-Dawn lifetime probe for GPUCompilationManager.
 *
 * This deliberately runs as its own process. A successful receipt proves that
 * one Dawn instance can service a queue of manager-owned async compilations,
 * drain, invalidate, destroy its device, and then exit without ProcessEvents
 * observing a collected native instance.
 */
import { writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  gpuCompilationManagerFor,
  invalidateGPUCompilationManager,
} from "../lib/core/gpu-compilation-manager";
import {
  createProcessRetainedDawnGPU,
  type NodeDawnProvider,
} from "../lib/harness/node-dawn-provider";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DEFAULT_DAWN_MODULE = `${ROOT}/node_modules/webgpu/index.js`;
const PIPELINE_COUNT = 64;

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.slice(2).find((value) => value.startsWith(prefix))
    ?.slice(prefix.length);
};

const nextEventLoopTurn = (): Promise<void> =>
  new Promise((resolve) => setImmediate(resolve));

async function main(): Promise<void> {
  const backend = argument("backend") ?? process.env.FLUID_WEBGPU_BACKEND ?? "metal";
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? DEFAULT_DAWN_MODULE;
  const outputPath = argument("out");
  await acquireWebGPUExclusiveLock(
    "dawn-acceptance",
    "tools/check-gpu-compilation-manager-dawn-lifecycle.ts",
  );

  let device: GPUDevice | undefined;
  let manager: ReturnType<typeof gpuCompilationManagerFor> | undefined;
  let drainedSnapshot: ReturnType<NonNullable<typeof manager>["snapshot"]> | undefined;
  const uncaptured: string[] = [];
  try {
    const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${backend}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error(`No Dawn adapter is available for backend ${backend}`);
    device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", (event) => {
      event.preventDefault();
      uncaptured.push(event.error.message);
    });
    device.pushErrorScope("validation");

    manager = gpuCompilationManagerFor(device);
    const module = manager.createShaderModule({
      label: "managed Dawn lifetime probe module",
      code: `
override SLOT: u32 = 0u;

@compute @workgroup_size(1)
fn main() {
  _ = SLOT;
}
`,
    });
    const pipelines = Array.from({ length: PIPELINE_COUNT }, (_, slot) =>
      manager!.compileComputePipeline({
        label: `managed Dawn lifetime probe pipeline ${slot}`,
        layout: "auto",
        compute: {
          module,
          entryPoint: "main",
          constants: { SLOT: slot },
        },
      }, { priority: slot === 0 ? "critical" : "background" }));
    await Promise.all(pipelines);
    await manager.whenIdle();

    const validation = await device.popErrorScope();
    if (validation) throw new Error(`validation failure: ${validation.message}`);
    if (uncaptured.length > 0) {
      throw new Error(`uncaptured validation failures: ${uncaptured.join(" | ")}`);
    }
    drainedSnapshot = manager.snapshot();
    if (drainedSnapshot.active !== 0 || drainedSnapshot.queued !== 0) {
      throw new Error(`manager did not drain: ${JSON.stringify(drainedSnapshot)}`);
    }
  } finally {
    if (device) {
      const compiler = manager ?? gpuCompilationManagerFor(device);
      invalidateGPUCompilationManager(device, "minimal Dawn lifecycle probe complete");
      await compiler.whenIdle();
      try { await device.queue.onSubmittedWorkDone(); } catch { /* Device fault is reported above. */ }
      device.destroy();
      await nextEventLoopTurn();
    }
    await releaseWebGPUExclusiveLock();
  }

  const receipt = {
    schema: "gpu-compilation-manager-dawn-lifecycle/v1",
    backend,
    pipelineCount: PIPELINE_COUNT,
    manager: drainedSnapshot,
    uncaptured,
    teardown: "manager-drained-device-destroyed",
    status: "pass",
  } as const;
  const json = `${JSON.stringify(receipt, null, 2)}\n`;
  if (outputPath) await writeFile(outputPath, json, "utf8");
  process.stdout.write(json);
}

await main();
