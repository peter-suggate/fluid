import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import { createPaperScenario } from "../lib/paper-scenarios";
import { initializeRigidBodies } from "../lib/rigid-body";
import { GPU_RIGID_RENDER_BYTES, GPU_RIGID_RENDER_FLOATS } from "../lib/webgpu-rigid-body";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("WebGPU did not expose an adapter");
const device = await adapter.requestDevice();
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

const scene = createPaperScenario("dam-break-boxes");
const dt = scene.numerics.fixedDt_s;
const target_s = Number(process.env.FLUID_TARGET_S ?? 0.3);
const bodies = initializeRigidBodies(scene.rigidBodies);
const initialCenters = bodies.map((body) => ({ ...body.position_m }));
const solver = octreeMethod.createSolver!(device, scene, "balanced", octreeMethod.presetFor("balanced"), undefined);

let submittedTime_s = 0;
while (submittedTime_s + 1e-9 < target_s) {
  submittedTime_s = Math.min(target_s, submittedTime_s + dt);
  assert.equal(solver.advanceTo(submittedTime_s, bodies), true, "resident fluid/rigid step should accept every fixed step");
}
await device.queue.onSubmittedWorkDone();
const info = await solver.readStats();
assert.ok(solver.rigidRenderBuffer, "solver must publish GPU-authored rigid render records");
const readback = device.createBuffer({ label: "test-only resident rigid snapshot", size: GPU_RIGID_RENDER_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(solver.rigidRenderBuffer!, 0, readback, 0, GPU_RIGID_RENDER_BYTES); device.queue.submit([encoder.finish()]);
await readback.mapAsync(GPUMapMode.READ);
const records = new Float32Array(readback.getMappedRange()).slice(); readback.unmap(); readback.destroy();
const bodyMotion_m = bodies.map((_body, index) => {
  const offset = index * GPU_RIGID_RENDER_FLOATS, initial = initialCenters[index];
  return Math.hypot(records[offset] - initial.x, records[offset + 1] - initial.y, records[offset + 2] - initial.z);
});
const result = { scenario: "octree-resident-rigid-feedback", target_s, dt, bodyMotion_m, pressurePasses: info.quadtreePressureIterationsUsed, nonFiniteVelocityCount: info.nonFiniteCount ?? 0, validationErrors };
console.log(JSON.stringify(result));
assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);
assert.ok(Math.max(...bodyMotion_m) > 1e-4, "at least one resident dynamic body must move");
assert.equal(info.quadtreePressureIterationsUsed, 32, "dynamic resident bodies must retain the accelerated pressure path");
assert.equal(info.nonFiniteCount ?? 0, 0, "fluid velocity must remain finite");
solver.destroy(); device.destroy();
