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
const device = await adapter!.requestDevice();
device.addEventListener("uncapturederror", (event: any) => console.error("UNCAPTURED:", event.error.message));

const scene = createSmokeScenario("dam-break-ui").scene;
const values = tallCellMethod.presetFor("balanced");
values.regularLayers = 24;
const bodies = initializeRigidBodies(scene.rigidBodies);
const solver: any = tallCellMethod.createSolver!(device, scene, "balanced", values);

async function readMax(texture: GPUTexture, label: string) {
  const { nx, storedNy, nz } = solver.info;
  const bytesPerRow = Math.ceil(nx * 16 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * storedNy * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: storedNy }, { width: nx, height: storedNy, depthOrArrayLayers: nz });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  let maximum = 0, count = 0;
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < storedNy; y += 1) {
    const row = new Float32Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + storedNy * z), nx * 4);
    for (let i = 0; i < nx * 4; i += 1) { const v = Math.abs(row[i]); if (v > 1e-9) count += 1; if (v > maximum) maximum = v; }
  }
  buffer.unmap(); buffer.destroy();
  console.log(label, "max=", maximum, "nonzero=", count);
}

for (let step = 1; step <= 2; step += 1) {
  solver.advanceTo(0.004 * step, bodies);
  await device.queue.onSubmittedWorkDone();
  console.log("--- after step", step);
  await readMax(solver.velocityA, "A");
  await readMax(solver.velocityB, "B");
  await readMax(solver.velocityC, "C(pre-projection)");
  await readMax(solver.velocityD, "D");
}
solver.destroy();
Reflect.deleteProperty(globalThis, "navigator");
// Appended: replicate the runner's reconstruction on a fresh solver.
const solver2: any = tallCellMethod.createSolver!(device, scene, "balanced", { ...tallCellMethod.presetFor("balanced"), regularLayers: 24 });
solver2.advanceTo(0.004, bodies);
await device.queue.onSubmittedWorkDone();
await solver2.readStats();
await device.queue.onSubmittedWorkDone();
{
  const { nx, storedNy, nz, ny } = solver2.info;
  const basesBytesPerRow = Math.ceil(nx * 4 / 256) * 256;
  const basesBuffer = device.createBuffer({ size: basesBytesPerRow * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture: solver2.columnBaseTexture }, { buffer: basesBuffer, bytesPerRow: basesBytesPerRow, rowsPerImage: nz }, { width: nx, height: nz });
  device.queue.submit([encoder.finish()]);
  await basesBuffer.mapAsync(GPUMapMode.READ);
  const basesRaw = new Uint8Array(basesBuffer.getMappedRange());
  const bases = new Float32Array(nx * nz);
  for (let z = 0; z < nz; z += 1) bases.set(new Float32Array(basesRaw.buffer, basesRaw.byteOffset + basesBytesPerRow * z, nx), nx * z);
  basesBuffer.unmap(); basesBuffer.destroy();
  console.log("bases: min", Math.min(...bases), "max", Math.max(...bases));

  const bytesPerRow = Math.ceil(nx * 16 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * storedNy * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder2 = device.createCommandEncoder();
  encoder2.copyTextureToBuffer({ texture: solver2.preProjectionVelocityTexture }, { buffer, bytesPerRow, rowsPerImage: storedNy }, { width: nx, height: storedNy, depthOrArrayLayers: nz });
  device.queue.submit([encoder2.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  const raw = new Float32Array(nx * storedNy * nz * 4);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < storedNy; y += 1) {
    const row = new Float32Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + storedNy * z), nx * 4);
    raw.set(row, nx * 4 * (y + storedNy * z));
  }
  buffer.unmap(); buffer.destroy();
  let rawMax = 0; for (const v of raw) rawMax = Math.max(rawMax, Math.abs(v));
  const packedAt = (x: number, py: number, z: number, axis: number) => raw[4 * (x + nx * (py + storedNy * z)) + axis];
  let reconstructedMax = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    for (let y = 0; y < ny; y += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        let value = 0;
        if (y < base && base > 0) {
          const t = Math.max(0, Math.min(1, y / Math.max(base - 1, 1)));
          value = (1 - t) * packedAt(x, 0, z, axis) + t * packedAt(x, 1, z, axis);
        } else {
          const py = 2 + y - base;
          if (py >= 2 && py < storedNy) value = packedAt(x, py, z, axis);
        }
        reconstructedMax = Math.max(reconstructedMax, Math.abs(value));
      }
    }
  }
  console.log("solver2 step1 C rawMax=", rawMax, "reconstructedMax=", reconstructedMax);
}
solver2.destroy();
device.destroy();
Reflect.deleteProperty(globalThis, "navigator");
