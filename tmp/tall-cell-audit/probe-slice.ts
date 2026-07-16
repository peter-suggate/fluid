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
scene.fluid.surfaceTension_N_m = 0;
const bodies = initializeRigidBodies(scene.rigidBodies);
const values = { ...tallCellMethod.presetFor("balanced"), regularLayers: 24 };
const solver: any = tallCellMethod.createSolver!(device, scene, "balanced", values);
const steps = Number(process.env.PROBE_STEPS ?? 10);
for (let step = 1; step <= steps; step += 1) solver.advanceTo(0.004 * step, bodies);
await device.queue.onSubmittedWorkDone();

const { nx, storedNy, nz } = solver.info;
async function readTexture(texture: GPUTexture, components: number) {
  const bytesPerRow = Math.ceil(nx * 4 * components / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * storedNy * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: storedNy }, { width: nx, height: storedNy, depthOrArrayLayers: nz });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  const out = new Float32Array(nx * storedNy * nz * components);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < storedNy; y += 1) {
    out.set(new Float32Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + storedNy * z), nx * components), nx * components * (y + storedNy * z));
  }
  buffer.unmap(); buffer.destroy();
  return out;
}
const velocity = await readTexture(solver.velocityTexture, 4);
const volume = await readTexture(solver.volumeTexture, 1);
const pressure = await readTexture((solver as any).pressureA, 1);

function fmt(v: number) { return (v >= 0 ? "+" : "") + v.toFixed(2); }
// Bottom endpoint plane (packed y=0): print v.z and v.x, alpha, pressure along z for a few x rows near the dam corner.
console.log(`--- after ${steps} steps: bottom-endpoint (packed y=0) slices`);
for (const x of [0, 1, 2, 10, 20, 28, 29, 30, 31]) {
  let rowVz = `x=${String(x).padStart(2)} v.z: `, rowVx = "      v.x: ", rowA = "      a  : ", rowP = "      p  : ";
  for (let z = 12; z < 28 && z < nz; z += 1) {
    const i = x + nx * (0 + storedNy * z);
    rowVz += fmt(velocity[4 * i + 2]) + " ";
    rowVx += fmt(velocity[4 * i + 0]) + " ";
    rowA += volume[i].toFixed(2).padStart(5) + " ";
    rowP += String(Math.round(pressure[i])).padStart(5) + " ";
  }
  console.log(rowVz); console.log(rowVx); console.log(rowA); console.log(rowP);
}
// Vertical profile of v.z through packed samples at the face column (x=0, z=19) and neighbors.
console.log("--- packed vertical profiles v.z (rows: packedY 0..25) at x=0");
for (const z of [17, 18, 19, 20, 21]) {
  let row = `z=${z}: `;
  for (let py = 0; py < storedNy; py += 1) row += fmt(velocity[4 * (0 + nx * (py + storedNy * z)) + 2]) + " ";
  console.log(row);
}
solver.destroy(); device.destroy();
Reflect.deleteProperty(globalThis, "navigator");
