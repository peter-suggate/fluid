// Dam-break-boxes gap-metric time series, optionally with rigid bodies
// removed, to separate refill-equilibrium drain from box displacement.
// Usage: WEBGPU_NODE_MODULE=... PROBE_BODIES=0|1 npx tsx tmp/tall-cell-audit/probe-boxes.ts
import { pathToFileURL } from "node:url";
import { tallCellMethod } from "../../lib/methods/tall-cell";
import { createPaperScenario } from "../../lib/paper-scenarios";
import { initializeRigidBodies, type RigidBodyState } from "../../lib/rigid-body";

const modulePath = process.env.WEBGPU_NODE_MODULE!;
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(o: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
const device = await adapter!.requestDevice();
device.addEventListener("uncapturederror", (event: any) => console.error("UNCAPTURED:", (event as any).error.message));

const scene = createPaperScenario("dam-break-boxes");
if (process.env.PROBE_BODIES === "0") scene.rigidBodies = [];
const values = { ...Object.fromEntries(tallCellMethod.params.map((p) => [p.key, p.default])), regularLayers: 12 } as Record<string, string | number>;
const solver: any = tallCellMethod.createSolver!(device, scene, "balanced", values, () => {});
const info = solver.info;
const dt = scene.numerics.maxDt_s;
const bodies: RigidBodyState[] = initializeRigidBodies(scene.rigidBodies);
console.log(`bodies=${bodies.length} grid=${info.nx}x${info.ny}x${info.nz} storedNy=${info.storedNy} dt=${dt}`);

async function read(texture: GPUTexture, w: number, h: number, d: number) {
  const bytesPerRow = Math.ceil(w * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  return { raw, rowFloats: bytesPerRow / 4 };
}

let t = 0;
for (let target = 0.5; target <= 5.0001; target += 0.5) {
  while (t < target - 1e-9) {
    t = Math.min(target, t + dt);
    solver.advanceTo(t, bodies);
  }
  await solver.readStats();
  await device.queue.onSubmittedWorkDone();
  const { nx, nz, storedNy } = info;
  const vol = await read(solver.volumeTexture, nx, storedNy, nz);
  const bas = await read(solver.columnBaseTexture, nx, nz, 1);
  let dry = 0, dryUnderWet = 0, minStore = 2, sumStore = 0, stores = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bas.raw[x + bas.rowFloats * z]);
    if (base < 2) continue;
    const bottom = Math.max(0, vol.raw[x + vol.rowFloats * (0 + storedNy * z)]);
    stores += 1; sumStore += Math.min(1, bottom); minStore = Math.min(minStore, bottom);
    if (bottom >= 0.5) continue;
    dry += 1;
    for (let py = 2; py < storedNy; py += 1) {
      if (vol.raw[x + vol.rowFloats * (py + storedNy * z)] >= 0.5) { dryUnderWet += 1; break; }
    }
  }
  console.log(`t=${t.toFixed(2)} dry=${dry} dryUnderWet=${dryUnderWet} meanStore=${(sumStore / Math.max(1, stores)).toFixed(3)} minStore=${minStore.toFixed(3)} relRes=${info.pressureRelativeResidual?.toExponential(1)} maxSpd=${info.maxSpeed_m_s?.toFixed(2)} flags=${info.stabilityFlags?.join("|") || "-"}`);
}
solver.destroy(); device.destroy();
