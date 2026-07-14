import { pathToFileURL } from "node:url";
import { createPaperScenario } from "../lib/paper-scenarios";
import { initializeRigidBodies } from "../lib/rigid-body";
import { tallCellMethod } from "../lib/methods/tall-cell";
import { uniformMethod } from "../lib/methods/uniform";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

for (const method of [tallCellMethod, uniformMethod]) {
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Dawn did not expose a Metal WebGPU adapter");
  const device = await adapter.requestDevice();
  let lost: GPUDeviceLostInfo | undefined;
  void device.lost.then((info) => { lost = info; });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  const instrumentedDevice = new Proxy(device, {
    get(target, property) {
      if (property === "createComputePipeline") return (descriptor: GPUComputePipelineDescriptor) => {
        const started = performance.now();
        const result = target.createComputePipeline(descriptor);
        console.log(JSON.stringify({ method: method.id, phase: "pipeline", entryPoint: descriptor.compute.entryPoint, elapsed_ms: Math.round(performance.now() - started) }));
        return result;
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    }
  }) as GPUDevice;
  const scene = createPaperScenario("hose-tank");
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const quality = "balanced" as const;
  const constructionStarted = performance.now();
  const solver = method.createSolver!(instrumentedDevice, scene, quality, method.presetFor(quality));
  const construction_ms = performance.now() - constructionStarted;
  console.log(JSON.stringify({ method: method.id, phase: "constructed", construction_ms: Math.round(construction_ms) }));
  const runStarted = performance.now();
  const target_s = 3;
  let steps = 0;
  while ((solver.info.simulatedTime_s ?? 0) + 1e-9 < target_s) {
    const requestedTime = Math.min(target_s, (steps + 1) * scene.numerics.maxDt_s);
    solver.advanceTo(requestedTime, bodies);
    steps += 1;
    if (steps % 30 === 0) await device.queue.onSubmittedWorkDone();
    if (steps % 180 === 0) console.log(JSON.stringify({ method: method.id, phase: "running", steps, simulatedTime_s: solver.info.simulatedTime_s }));
    if (lost) throw new Error(`${method.id} device lost: ${lost.message || lost.reason}`);
  }
  await device.queue.onSubmittedWorkDone();
  const info = await solver.readStats();
  await device.queue.onSubmittedWorkDone();
  const runtime_ms = performance.now() - runStarted;
  console.log(JSON.stringify({
    method: method.id,
    adapter: adapter.info,
    construction_ms: Math.round(construction_ms),
    runtime_ms: Math.round(runtime_ms),
    steps,
    simulatedTime_s: info.simulatedTime_s,
    grid: [info.nx, info.storedNy, info.nz],
    encodedSteps: info.encodedSteps,
    initialVolumeCellSum: info.initialVolumeCellSum,
    volumeCellSum: info.volumeCellSum,
    representedVolumeCellSum: info.representedVolumeCellSum,
    volumeDrift: info.volumeDrift,
    representedVolumeDrift: info.representedVolumeDrift,
    maxSpeed_m_s: info.maxSpeed_m_s,
    nonFiniteCount: info.nonFiniteCount,
    stabilityFlags: info.stabilityFlags,
    validationErrors
  }));
  solver.destroy();
  device.destroy();
}

Reflect.deleteProperty(globalThis, "navigator");
