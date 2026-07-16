import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../../lib/methods/tall-cell";
import { createSmokeScenario } from "../../tools/webgpu-smoke-scenarios";
import { initializeRigidBodies } from "../../lib/rigid-body";

const modulePath = process.env.WEBGPU_NODE_MODULE!;
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
const requiredFeatures: GPUFeatureName[] = process.env.PROBE_TS === '1' && adapter!.features.has('timestamp-query') ? ['timestamp-query'] : [];
const device = await adapter!.requestDevice({ requiredFeatures });
device.addEventListener("uncapturederror", (event: any) => console.error("UNCAPTURED:", event.error.message));
device.pushErrorScope("validation");

const scene = createSmokeScenario((process.env.PROBE_SCENE ?? "sphere-jet") as any).scene;
const bodies = initializeRigidBodies(scene.rigidBodies);
const solver: any = tallCellMethod.createSolver!(device, scene, "balanced", { ...tallCellMethod.presetFor("balanced"), ...(process.env.PROBE_CYCLES ? { pressureCycles: Number(process.env.PROBE_CYCLES) } : {}), ...(process.env.PROBE_LAYERS ? { regularLayers: Number(process.env.PROBE_LAYERS) } : {}) });
console.log("gridKind", solver.info.gridKind, "dims", solver.info.nx, solver.info.storedNy, solver.info.nz);
for (let step = 1; step <= Number(process.env.PROBE_STEPS ?? 3); step += 1) {
  device.pushErrorScope("validation");
  const accepted = solver.advanceTo(scene.numerics.maxDt_s * step, process.env.PROBE_NOBODIES === "1" ? [] : bodies);
  const stepError = await device.popErrorScope();
  console.log("step", step, "accepted", accepted, "error:", stepError ? stepError.message.slice(0, 300) : "none");
}
await device.queue.onSubmittedWorkDone();
const err = await device.popErrorScope();
if (err) console.error("SCOPE ERROR:", err.message.slice(0, 400));

const { nx, storedNy, nz } = solver.info;
async function readMax(texture: GPUTexture, comps: number, label: string) {
  const bytesPerRow = Math.ceil(nx * 4 * comps / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * storedNy * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: storedNy }, { width: nx, height: storedNy, depthOrArrayLayers: nz });
  device.queue.submit([e.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  let maximum = 0, count = 0, sum = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < storedNy; y += 1) {
    const row = new Float32Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + storedNy * z), nx * comps);
    for (const v of row) { const a = Math.abs(v); sum += a; if (a > 1e-9) count += 1; if (a > maximum) maximum = a; }
  }
  buffer.unmap(); buffer.destroy();
  console.log(label, "max", maximum.toFixed(5), "nonzero", count, "sum", sum.toFixed(2));
}
await readMax((solver as any).velocityA, 4, "velocityA");
await readMax((solver as any).velocityD, 4, "velocityD");
await readMax((solver as any).pressureA, 1, "pressureA");
await readMax(solver.volumeTexture, 1, "volumeA");
const stats = await solver.readStats();
console.log("stats maxSpeed", stats.maxSpeed_m_s, "volumeCellSum", stats.volumeCellSum, "nonFinite", stats.nonFiniteCount);
solver.destroy(); device.destroy();
Reflect.deleteProperty(globalThis, "navigator");
