// Check whether dam-break water is pressing against the band ceiling at
// t=0.416s: per column, report band-top alpha and highest wet worldY.
import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../../lib/methods/tall-cell";
import { createSmokeScenario } from "../../tools/webgpu-smoke-scenarios";
import { initializeRigidBodies } from "../../lib/rigid-body";

const modulePath = process.env.WEBGPU_NODE_MODULE!;
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(o: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

const target = Number(process.env.FLUID_TARGET_S ?? 0.416);
const scenario = createSmokeScenario("dam-break-ui");
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
const device = await adapter!.requestDevice({ requiredLimits: { maxTextureDimension3D: Math.min(2048, adapter!.limits.maxTextureDimension3D) } });
const values = Object.fromEntries(tallCellMethod.params.map((p) => [p.key, p.default])) as Record<string, string | number>;
const solver = tallCellMethod.createSolver!(device, scenario.scene, "balanced", values, () => {}) as import("../../lib/webgpu-eulerian").WebGPUEulerianSolver;
const bodies = initializeRigidBodies(scenario.scene.rigidBodies);
const dt = scenario.scene.numerics.maxDt_s;
let t = 0;
while (t < target - 1e-9) { t = Math.min(target, t + dt); solver.advanceTo(t, bodies); await solver.readStats(); }
await device.queue.onSubmittedWorkDone();

async function readTexture3D(texture: GPUTexture, w: number, h: number, d: number) {
  const bytesPerRow = Math.ceil(w * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(buffer.getMappedRange().slice(0));
  const out = new Float32Array(w * h * d);
  const rowFloats = bytesPerRow / 4;
  for (let z = 0; z < d; z += 1) for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) out[x + w * (y + h * z)] = raw[x + rowFloats * (y + h * z)];
  buffer.destroy();
  return out;
}

const info = solver.info;
const packed = await readTexture3D(solver.volumeTexture, info.nx, info.storedNy, info.nz);
const bases = await readTexture3D(solver.columnBaseTexture, info.nx, info.nz, 1);
const layers = info.storedNy - 2;
let ceilingWet = 0, total = 0, maxTopAlpha = 0;
const hist: number[] = [];
const examples: string[] = [];
for (let z = 0; z < info.nz; z += 1) for (let x = 0; x < info.nx; x += 1) {
  const base = Math.round(bases[x + info.nx * z]);
  total += 1;
  const topAlpha = packed[x + info.nx * (info.storedNy - 1 + info.storedNy * z)];
  maxTopAlpha = Math.max(maxTopAlpha, topAlpha);
  if (topAlpha >= 0.5) {
    ceilingWet += 1;
    if (examples.length < 12) examples.push(`x=${x} z=${z} base=${base} topAlpha=${topAlpha.toFixed(3)} (bandTop worldY=${base + layers - 1})`);
  }
  let highWet = -1;
  for (let py = info.storedNy - 1; py >= 2; py -= 1) {
    if (packed[x + info.nx * (py + info.storedNy * z)] >= 0.5) { highWet = base + py - 2; break; }
  }
  hist.push(highWet);
}
const over = (y: number) => hist.filter((h) => h >= y).length;
console.log(`t=${target}s columns=${total} ceilingWet(top band cell alpha>=0.5)=${ceilingWet} maxTopAlpha=${maxTopAlpha.toFixed(3)}`);
console.log(`columns with highest wet >= 20: ${over(20)}, >= 23: ${over(23)}, >= 25 (band top): ${over(25)}`);
for (const e of examples) console.log("  " + e);
// also: total alpha stored in the top two band rows (mass jammed at ceiling)
let jam = 0;
for (let z = 0; z < info.nz; z += 1) for (let x = 0; x < info.nx; x += 1)
  for (let py = info.storedNy - 2; py < info.storedNy; py += 1) jam += packed[x + info.nx * (py + info.storedNy * z)];
console.log(`alpha sum in top two band rows: ${jam.toFixed(2)} cells`);
console.log(`volumeDrift=${(solver.info.volumeDrift! * 100).toFixed(2)}% volumeCellSum=${solver.info.volumeCellSum?.toFixed(1)} initial=${solver.info.initialVolumeCellSum?.toFixed(1)}`);
// Wall column profiles: packed alphas bottom→top for a few (x, z=0) columns.
for (const x of [0, 5, 30, 55]) {
  const z = 0;
  const base = Math.round(bases[x + info.nx * z]);
  const store = packed[x + info.nx * info.storedNy * z];
  const band: string[] = [];
  for (let py = 2; py < info.storedNy; py += 1) band.push(packed[x + info.nx * (py + info.storedNy * z)].toFixed(2));
  console.log(`x=${x} z=0 base=${base} store=${store.toFixed(3)} band(y=${base}..${base + layers - 1}): ${band.join(" ")}`);
}
solver.destroy(); device.destroy();
