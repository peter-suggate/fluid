// Dump a mid-z slice of the tall solver's packed volume state at a given
// time: per-column base, store average, and band-cell alphas, rendered as
// ASCII so classification (wet >= 0.5) can be inspected directly.
// Usage: WEBGPU_NODE_MODULE=... FLUID_TARGET_S=0.224 npx tsx tools/dump-tall-slice.ts
import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../lib/methods/tall-cell";
import { createSmokeScenario } from "./webgpu-smoke-scenarios";
import { initializeRigidBodies } from "../lib/rigid-body";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE");
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(o: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

const target = Number(process.env.FLUID_TARGET_S ?? 0.224);
const scenario = createSmokeScenario("dam-break-ui");
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no adapter");
const device = await adapter.requestDevice({ requiredLimits: { maxTextureDimension3D: Math.min(2048, adapter.limits.maxTextureDimension3D) } });
const values = Object.fromEntries(tallCellMethod.params.map((p) => [p.key, p.default])) as Record<string, string | number>;
const solver = tallCellMethod.createSolver(device, scenario.scene, "balanced", values, () => {}) as import("../lib/webgpu-eulerian").WebGPUEulerianSolver;
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
async function readTexture2D(texture: GPUTexture, w: number, h: number) {
  const field = await readTexture3D(texture, w, h, 1);
  return field;
}

const info = solver.info;
const packed = await readTexture3D(solver.volumeTexture, info.nx, info.storedNy, info.nz);
const bases = await readTexture2D(solver.columnBaseTexture, info.nx, info.nz);
const z = Math.floor(info.nz / 2);
const rows: string[] = [];
for (let y = info.ny - 1; y >= 0; y -= 1) {
  let row = "";
  for (let x = 0; x < info.nx; x += 1) {
    const base = Math.round(bases[x + info.nx * z]);
    let ch = " ";
    if (y < base && base > 0) {
      const avg = packed[x + info.nx * info.storedNy * z];
      ch = avg >= 0.5 ? "T" : avg >= 0.05 ? "t" : ".";
    } else {
      const packedY = 2 + y - base;
      if (packedY >= 2 && packedY < info.storedNy) {
        const a = packed[x + info.nx * (packedY + info.storedNy * z)];
        ch = a >= 0.5 ? "#" : a >= 0.05 ? "+" : "-";
      }
    }
    row += ch;
  }
  rows.push(`${String(y).padStart(3)} ${row}`);
}
console.log(`t=${t.toFixed(3)}s z=${z}  T/t/.=tall store wet/partial/dry  #/+/-=band wet/partial/dry  (blank=unrepresented)`);
console.log(rows.join("\n"));
// Column diagnostics under the deep region: store average and band-bottom alpha
let dryUnderWet = 0, wetStores = 0, total = 0;
const examples: string[] = [];
for (let zz = 0; zz < info.nz; zz += 1) for (let x = 0; x < info.nx; x += 1) {
  const base = Math.round(bases[x + info.nx * zz]);
  if (base <= 0) continue;
  total += 1;
  const avg = packed[x + info.nx * info.storedNy * zz];
  const bandBottom = packed[x + info.nx * (2 + info.storedNy * zz)];
  if (avg >= 0.5) wetStores += 1;
  else if (bandBottom >= 0.5) { dryUnderWet += 1; if (examples.length < 10) examples.push(`x=${x} z=${zz} base=${base} avg=${avg.toFixed(3)} bandBottom=${bandBottom.toFixed(3)}`); }
}
console.log(`columns=${total} wetStores=${wetStores} dryUnderWetBand=${dryUnderWet}`);
// Locate kinetic energy: alpha-weighted |v|^2 split by sample kind and base height.
{
  const readVec4 = async (texture: GPUTexture, w: number, h: number, d: number) => {
    const bytesPerRow = Math.ceil(w * 16 / 256) * 256;
    const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(buffer.getMappedRange().slice(0));
    buffer.destroy();
    const rowFloats = bytesPerRow / 4;
    return { raw, rowFloats };
  };
  const velocity = await readVec4((solver as unknown as { velocityTexture: GPUTexture }).velocityTexture, info.nx, info.storedNy, info.nz);
  const vel2 = (x: number, py: number, zz: number) => {
    const o = 4 * x + velocity.rowFloats * (py + info.storedNy * zz);
    return velocity.raw[o] * velocity.raw[o] + velocity.raw[o + 1] * velocity.raw[o + 1] + velocity.raw[o + 2] * velocity.raw[o + 2];
  };
  let keStoreShallow = 0, keStoreDeep = 0, keBand = 0;
  let maxStore = 0, maxStoreAt = "";
  for (let zz = 0; zz < info.nz; zz += 1) for (let x = 0; x < info.nx; x += 1) {
    const base = Math.round(bases[x + info.nx * zz]);
    for (let py = 0; py < info.storedNy; py += 1) {
      const i = x + info.nx * (py + info.storedNy * zz);
      const a = Math.max(0, Math.min(1, packed[i]));
      const speed2 = vel2(x, py, zz);
      if (py < 2) {
        const weight = a * (py === 0 ? Math.max(base - 1, 1) : 1);
        if (base <= 3) keStoreShallow += weight * speed2; else keStoreDeep += weight * speed2;
        if (Math.sqrt(speed2) > maxStore && a >= 0.5) { maxStore = Math.sqrt(speed2); maxStoreAt = `x=${x} z=${zz} py=${py} base=${base}`; }
      } else keBand += a * speed2;
    }
  }
  console.log(`KEproxy store(base<=3)=${keStoreShallow.toFixed(3)} store(base>3)=${keStoreDeep.toFixed(3)} band=${keBand.toFixed(3)} maxStoreSpeed=${maxStore.toFixed(2)} @ ${maxStoreAt}`);
}
for (const e of examples) console.log("  " + e);
solver.destroy(); device.destroy();
Reflect.deleteProperty(globalThis, "navigator");
